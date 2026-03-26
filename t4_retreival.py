import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import pipeline as hf_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import librosa
import dotenv
dotenv.load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
INDEX_FILENAME = "audio_index.csv"

EMOTION_LABEL_MAP = {
    "ang": "angry", "hap": "happy", "neu": "neutral",
    "sad": "sad", "fea": "fearful", "dis": "disgusted",
    "cal": "calm", "exc": "excited",
}

_asr = None
_emotion = None
_embedder = None
_llm = None

# -------------------------------
# MODELS
# -------------------------------
def get_asr():
    global _asr
    if _asr is None:
        print("⏳ Loading Whisper ASR …")
        _asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small")
    return _asr

def get_emotion():
    global _emotion
    if _emotion is None:
        print("⏳ Loading emotion classifier …")
        _emotion = hf_pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    return _emotion

def get_embedder():
    global _embedder
    if _embedder is None:
        print("⏳ Loading sentence embedder …")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
    return _llm

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(audio_path: str) -> dict:
    path = Path(audio_path)

    y, sr = librosa.load(str(path), sr=16000, mono=True)
    y = y / np.max(np.abs(y))  # normalize
    duration = round(librosa.get_duration(y=y, sr=sr), 2)

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    mean_energy = float(np.mean(rms))

    # Pitch (F0)
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        mean_pitch = float(np.nanmean(f0)) if f0 is not None else 0
    except:
        mean_pitch = 0

    # ASR transcription
    asr_result = get_asr()(str(path), generate_kwargs={"language": "en"})
    transcription = asr_result.get("text", "").strip()

    # Emotion / Tone classification
    emotion_results = get_emotion()(str(path), top_k=3)
    top_emotion = emotion_results[0]["label"].lower()
    top_emotion = EMOTION_LABEL_MAP.get(top_emotion, top_emotion)

    emotion_scores = {
        EMOTION_LABEL_MAP.get(r["label"].lower(), r["label"].lower()): round(r["score"], 3)
        for r in emotion_results
    }

    return {
        "file": str(path),
        "filename": path.name,
        "duration": duration,
        "energy": mean_energy,
        "pitch": mean_pitch,
        "transcription": transcription,
        "emotion": top_emotion,
        "emotion_scores": json.dumps(emotion_scores),
    }

# -------------------------------
# CSV INDEXING
# -------------------------------
def build_index(folder: str, index_path: str):
    folder_path = Path(folder)
    audio_files = [f for f in folder_path.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in '{folder}'")

    print(f"\n📂 Indexing {len(audio_files)} files...\n")

    # Load models once
    get_asr()
    get_emotion()
    get_embedder()

    rows = []
    for i, af in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {af.name}")
        try:
            rows.append(extract_features(str(af)))
        except Exception as e:
            print(f"⚠ Skipped {af.name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(index_path, index=False)
    print(f"\n✅ CSV saved → {index_path}\n")
    return rows

def load_or_build_index(folder: str):
    index_path = Path(folder) / INDEX_FILENAME
    if index_path.exists():
        print(f"✅ Loading existing CSV → {index_path}")
        df = pd.read_csv(index_path)
        df["emotion_scores"] = df["emotion_scores"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {}
        )
        return df.to_dict(orient="records")
    return build_index(folder, str(index_path))

# -------------------------------
# QUERY PARSING (LLM)
# -------------------------------
def parse_query(user_query: str):
    prompt = f"""
Extract structured search parameters from this query.

Return ONLY JSON with:
emotion, topic_keywords, min_duration, max_duration, min_energy, max_energy, min_pitch, max_pitch, semantic_query

Query: "{user_query}"
"""
    response = get_llm().invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].replace("json", "").strip()
    try:
        return json.loads(raw)
    except:
        return {
            "emotion": None,
            "topic_keywords": user_query.split(),
            "min_duration": None,
            "max_duration": None,
            "min_energy": None,
            "max_energy": None,
            "min_pitch": None,
            "max_pitch": None,
            "semantic_query": user_query,
        }

# -------------------------------
# SEMANTIC SEARCH + FILTERING
# -------------------------------
def make_document_text(entry):
    return (
        f"emotion: {entry['emotion']}. "
        f"energy: {entry.get('energy',0):.3f}. "
        f"pitch: {entry.get('pitch',0):.2f}. "
        f"text: {entry['transcription']}. "
        f"duration: {entry['duration']}"
    )

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def filter_and_rank(index, params, top_k=10):
    candidates = index

    # Duration filter
    if params.get("min_duration") is not None:
        candidates = [c for c in candidates if c["duration"] >= params["min_duration"]]
    if params.get("max_duration") is not None:
        candidates = [c for c in candidates if c["duration"] <= params["max_duration"]]

    # Energy filter
    if params.get("min_energy") is not None:
        candidates = [c for c in candidates if c.get("energy",0) >= params["min_energy"]]
    if params.get("max_energy") is not None:
        candidates = [c for c in candidates if c.get("energy",0) <= params["max_energy"]]

    # Pitch filter
    if params.get("min_pitch") is not None:
        candidates = [c for c in candidates if c.get("pitch") is not None and c["pitch"] >= params["min_pitch"]]
    if params.get("max_pitch") is not None:
        candidates = [c for c in candidates if c.get("pitch") is not None and c["pitch"] <= params["max_pitch"]]

    if not candidates:
        return []

    # Semantic search using embeddings
    embedder = get_embedder()
    query_emb = embedder.encode(params.get("semantic_query",""), normalize_embeddings=True)
    docs = [make_document_text(c) for c in candidates]
    doc_embs = embedder.encode(docs, normalize_embeddings=True)

    results = []
    for c, emb in zip(candidates, doc_embs):
        score = cosine_similarity(query_emb, emb)
        if params.get("emotion") and c["emotion"] == params["emotion"]:
            score += 0.15
        c["similarity_score"] = round(score, 4)
        results.append(c)

    return sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]

# -------------------------------
# LLM RERANK
# -------------------------------
def llm_rerank(query, candidates):
    if not candidates:
        return []
    text = "\n".join(
        f'{i+1}. {c["filename"]} | {c["emotion"]} | {c["duration"]}s | {c["transcription"]}'
        for i, c in enumerate(candidates)
    )
    prompt = f"""
Re-rank these for query: "{query}"

{text}

Return JSON list with rank, filename, reason
"""
    response = get_llm().invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].replace("json", "").strip()
    try:
        ranking = json.loads(raw)
    except:
        return candidates[:5]
    lookup = {c["filename"]: c for c in candidates}
    final = []
    for r in ranking:
        c = lookup.get(r["filename"], {})
        final.append({**c, "rank": r["rank"], "reason": r["reason"]})
    return final[:5]

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def semantic_audio_query(query, folder, rebuild=False):
    if rebuild:
        (Path(folder) / INDEX_FILENAME).unlink(missing_ok=True)
    index = load_or_build_index(folder)
    params = parse_query(query)
    candidates = filter_and_rank(index, params)
    return llm_rerank(query, candidates)

# -------------------------------
# PRINT
# -------------------------------
def pretty_print(results):
    print("\n" + "="*50)
    print("TOP RESULTS")
    print("="*50)
    for r in results:
        print(f"\n#{r.get('rank')} {r['filename']}")
        print("Emotion:", r["emotion"])
        print("Duration:", r["duration"])
        print("Energy:", round(r.get("energy",0), 3))
        print("Pitch:", round(r.get("pitch",0), 2))
        print("Score:", r.get("similarity_score"))
        print("Text:", r["transcription"])
        print("Why:", r.get("reason"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    results = semantic_audio_query(query=args.query, folder=args.folder, rebuild=args.rebuild)
    pretty_print(results)