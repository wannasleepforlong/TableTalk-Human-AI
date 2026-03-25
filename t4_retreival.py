
import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import pipeline as hf_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import librosa

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
INDEX_FILENAME   = "audio_index.json"

EMOTION_LABEL_MAP = {          # normalise wav2vec2 labels → plain English
    "ang": "angry",  "hap": "happy", "neu": "neutral",
    "sad": "sad",    "fea": "fearful", "dis": "disgusted",
    "cal": "calm",   "exc": "excited",
}

_asr        = None
_emotion    = None
_embedder   = None
_llm        = None


def get_asr():
    global _asr
    if _asr is None:
        print("⏳  Loading Whisper ASR …")
        _asr = hf_pipeline("automatic-speech-recognition",
                            model="openai/whisper-small")
    return _asr


def get_emotion():
    global _emotion
    if _emotion is None:
        print("⏳  Loading emotion classifier …")
        _emotion = hf_pipeline("audio-classification",
                                model="superb/wav2vec2-base-superb-er")
    return _emotion


def get_embedder():
    global _embedder
    if _embedder is None:
        print("⏳  Loading sentence embedder …")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=GEMINI_API_KEY,
            temperature=0.3,
        )
    return _llm


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def extract_features(audio_path: str) -> dict:
    """Return a dict with transcription, top emotion, duration for one file."""
    path = Path(audio_path)

    # Duration
    y, sr = librosa.load(str(path), sr=None, mono=True)
    duration = round(librosa.get_duration(y=y, sr=sr), 2)

    # Transcription
    asr_result = get_asr()(str(path), generate_kwargs={"language": "en"})
    transcription = asr_result["text"].strip()

    # Emotion — top label
    emotion_results = get_emotion()(str(path), top_k=3)
    top_emotion = emotion_results[0]["label"].lower()
    # Map short codes if present
    top_emotion = EMOTION_LABEL_MAP.get(top_emotion, top_emotion)
    emotion_scores = {
        EMOTION_LABEL_MAP.get(r["label"].lower(), r["label"].lower()): round(r["score"], 3)
        for r in emotion_results
    }

    return {
        "file":          str(path),
        "filename":      path.name,
        "duration":      duration,
        "transcription": transcription,
        "emotion":       top_emotion,
        "emotion_scores": emotion_scores,
    }


def build_index(folder: str, index_path: str) -> list[dict]:
    """Extract features for every audio file in folder, cache to JSON."""
    folder_path = Path(folder)
    audio_files = [
        f for f in folder_path.iterdir()
        if f.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in '{folder}'")

    print(f"\n📂  Indexing {len(audio_files)} audio files in '{folder}' …\n")
    index = []

    for i, af in enumerate(audio_files, 1):
        print(f"  [{i}/{len(audio_files)}] {af.name}")
        try:
            features = extract_features(str(af))
            index.append(features)
        except Exception as e:
            print(f"    ⚠  Skipped ({e})")

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n✅  Index saved → {index_path}\n")
    return index


def load_or_build_index(folder: str) -> list[dict]:
    index_path = Path(folder) / INDEX_FILENAME
    if index_path.exists():
        print(f"✅  Loading cached index from {index_path}")
        with open(index_path) as f:
            return json.load(f)
    return build_index(folder, str(index_path))


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — QUERY PARSING
# ════════════════════════════════════════════════════════════════════════════

def parse_query(user_query: str) -> dict:
    """Ask LLM to extract structured parameters from the natural language query."""
    prompt = f"""
You are a query parser for an audio search engine.
Extract search parameters from the user's natural language query.

Return ONLY valid JSON with these keys:
  - "emotion"       : string or null  (e.g. "happy", "sad", "angry", "neutral")
  - "topic_keywords": list of strings (key content words to match transcription)
  - "min_duration"  : number or null  (seconds)
  - "max_duration"  : number or null  (seconds)
  - "semantic_query": string          (a clean sentence summarising what the user wants,
                                       used for embedding similarity)

User query: "{user_query}"

JSON:"""

    response = get_llm().invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback — treat entire query as semantic query
        params = {
            "emotion": None,
            "topic_keywords": user_query.split(),
            "min_duration": None,
            "max_duration": None,
            "semantic_query": user_query,
        }

    print(f"\n🔍  Parsed query parameters:\n{json.dumps(params, indent=2)}\n")
    return params


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — COSINE SIMILARITY MATCHING
# ════════════════════════════════════════════════════════════════════════════

def make_document_text(entry: dict) -> str:
    """Combine metadata into a single text blob for embedding."""
    return (
        f"emotion: {entry['emotion']}. "
        f"transcription: {entry['transcription']}. "
        f"duration: {entry['duration']} seconds."
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def filter_and_rank(index: list[dict], params: dict, top_k: int = 10) -> list[dict]:
    """
    1. Hard-filter by duration and emotion (if specified).
    2. Cosine-rank remaining by semantic query.
    """
    candidates = index

    # Hard filter: duration
    if params.get("min_duration") is not None:
        candidates = [c for c in candidates if c["duration"] >= params["min_duration"]]
    if params.get("max_duration") is not None:
        candidates = [c for c in candidates if c["duration"] <= params["max_duration"]]

    # Soft filter: emotion bonus (don't hard-exclude — let similarity decide)
    # We'll encode emotion preference into the semantic query instead.

    if not candidates:
        return []

    embedder = get_embedder()
    query_emb = embedder.encode(params["semantic_query"], normalize_embeddings=True)

    doc_texts = [make_document_text(c) for c in candidates]
    doc_embs  = embedder.encode(doc_texts, normalize_embeddings=True, batch_size=32)

    scored = []
    for entry, emb in zip(candidates, doc_embs):
        score = cosine_similarity(query_emb, emb)

        # Boost score if emotion matches
        if params.get("emotion") and entry["emotion"] == params["emotion"]:
            score += 0.15

        scored.append({**entry, "similarity_score": round(score, 4)})

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:top_k]


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — LLM RERANKING + EXPLANATION
# ════════════════════════════════════════════════════════════════════════════

def llm_rerank(user_query: str, candidates: list[dict]) -> list[dict]:
    """Send top candidates to LLM for final reranking and explanation."""
    if not candidates:
        return []

    candidate_lines = "\n".join(
        f'{i+1}. File: {c["filename"]} | Emotion: {c["emotion"]} | '
        f'Duration: {c["duration"]}s | Similarity: {c["similarity_score"]} | '
        f'Transcription: "{c["transcription"]}"'
        for i, c in enumerate(candidates)
    )

    prompt = f"""
You are an intelligent audio search assistant.
A user searched for: "{user_query}"

Below are the top candidate audio files ranked by semantic similarity.
Your task:
1. Re-rank them based on how well they match the user's intent.
2. Return ONLY valid JSON: a list of objects, each with:
   - "rank"       : integer (1 = best)
   - "filename"   : string
   - "reason"     : one sentence explaining why this file matches

Candidates:
{candidate_lines}

JSON:"""

    response = get_llm().invoke(prompt)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        rankings = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return as-is with no reranking
        return [
            {**c, "rank": i+1, "reason": "Ranked by cosine similarity."}
            for i, c in enumerate(candidates[:5])
        ]

    # Merge LLM ranks back with full metadata
    filename_to_meta = {c["filename"]: c for c in candidates}
    results = []
    for r in sorted(rankings, key=lambda x: x.get("rank", 99)):
        fname = r.get("filename", "")
        meta  = filename_to_meta.get(fname, {})
        results.append({
            "rank":              r.get("rank"),
            "filename":         fname,
            "file":             meta.get("file", fname),
            "emotion":          meta.get("emotion"),
            "duration":         meta.get("duration"),
            "transcription":    meta.get("transcription"),
            "similarity_score": meta.get("similarity_score"),
            "reason":           r.get("reason", ""),
        })

    return results[:5]


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def semantic_audio_query(
    query: str,
    folder: str,
    rebuild_index: bool = False,
) -> list[dict]:
    """
    Main entry point.

    Args:
        query:         Natural language query, e.g. "happy audio about greetings"
        folder:        Path to audio folder
        rebuild_index: Force re-extraction even if index.json exists

    Returns:
        List of top-5 dicts, each with: rank, filename, emotion, duration,
        transcription, similarity_score, reason
    """
    # 1. Index
    if rebuild_index:
        index_path = Path(folder) / INDEX_FILENAME
        index_path.unlink(missing_ok=True)
    index = load_or_build_index(folder)

    # 2. Parse query
    params = parse_query(query)

    # 3. Cosine similarity — get top 10 for LLM reranking
    top_candidates = filter_and_rank(index, params, top_k=10)

    if not top_candidates:
        print("⚠  No candidates found after filtering.")
        return []

    # 4. LLM rerank → final top 5
    results = llm_rerank(query, top_candidates)

    return results



def pretty_print(results: list[dict]) -> None:
    print("\n" + "═" * 60)
    print("  🎧  TOP RESULTS")
    print("═" * 60)
    for r in results:
        print(f"\n  #{r['rank']}  {r['filename']}")
        print(f"      Emotion   : {r['emotion']}")
        print(f"      Duration  : {r['duration']}s")
        print(f"      Similarity: {r['similarity_score']}")
        print(f"      Text      : \"{r['transcription']}\"")
        print(f"      Why       : {r['reason']}")
    print("\n" + "═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Audio Query System")
    parser.add_argument("--folder",  required=True,  help="Path to audio folder (e.g. CREMA-D)")
    parser.add_argument("--query",   required=True,  help="Natural language query")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    args = parser.parse_args()

    results = semantic_audio_query(
        query=args.query,
        folder=args.folder,
        rebuild_index=args.rebuild,
    )

    pretty_print(results)