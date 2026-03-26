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

GEMINI_API_KEY   = os.getenv("GOOGLE_API_KEY")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
INDEX_FILENAME   = "audio_index.csv"

EMOTION_LABEL_MAP = {
    "ang": "angry", "hap": "happy", "neu": "neutral",
    "sad": "sad",   "fea": "fearful", "dis": "disgusted",
    "cal": "calm",  "exc": "excited",
}

# ──────────────────────────────────────────────
# LAZY MODEL SINGLETONS
# ──────────────────────────────────────────────
_asr      = None
_emotion  = None
_embedder = None
_llm      = None

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


# ──────────────────────────────────────────────
# FEATURE EXTRACTION (used when building index)
# ──────────────────────────────────────────────
def extract_features(audio_path: str) -> dict:
    path = Path(audio_path)
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    duration = round(librosa.get_duration(y=y, sr=sr), 2)

    rms = librosa.feature.rms(y=y)[0]
    mean_energy = float(np.mean(rms))

    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        mean_pitch = float(np.nanmean(f0)) if f0 is not None else 0.0
    except Exception:
        mean_pitch = 0.0

    asr_result   = get_asr()(str(path), generate_kwargs={"language": "en"})
    transcription = asr_result.get("text", "").strip()

    emotion_results = get_emotion()(str(path), top_k=5)
    top_emotion = EMOTION_LABEL_MAP.get(emotion_results[0]["label"].lower(), emotion_results[0]["label"].lower())
    emotion_scores = {
        EMOTION_LABEL_MAP.get(r["label"].lower(), r["label"].lower()): round(r["score"], 4)
        for r in emotion_results
    }

    return {
        "file":          str(path),
        "filename":      path.name,
        "duration":      duration,
        "energy":        mean_energy,
        "pitch":         mean_pitch,
        "transcription": transcription,
        "emotion":       top_emotion,
        "emotion_scores": json.dumps(emotion_scores),
    }


# ──────────────────────────────────────────────
# CSV INDEX
# ──────────────────────────────────────────────
def build_index(folder: str, index_path: str) -> list[dict]:
    folder_path = Path(folder)
    audio_files = [f for f in folder_path.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in '{folder}'")

    print(f"\n📂 Indexing {len(audio_files)} file(s)…\n")
    asr_pipe     = get_asr()
    emotion_pipe = get_emotion()
    rows = []
    for i, af in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {af.name}")
        try:
            rows.append(extract_features(str(af)))
        except Exception as e:
            print(f"  ⚠ Skipped: {e}")

    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"\n✅ Index saved → {index_path}\n")
    return rows


def load_or_build_index(folder: str) -> list[dict]:
    index_path = Path(folder) / INDEX_FILENAME
    if index_path.exists():
        print(f"✅ Loading existing index → {index_path}")
        df = pd.read_csv(index_path)
        df["emotion_scores"] = df["emotion_scores"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {}
        )
        return df.to_dict(orient="records")
    return build_index(folder, str(index_path))


# ──────────────────────────────────────────────
# STEP 1 — LLM QUERY PARSING
# ──────────────────────────────────────────────
_PARSE_PROMPT = """You are an audio retrieval assistant. Parse the user query into structured JSON filters.

Return ONLY valid JSON with these exact keys (use null for unknowns):
{{
  "emotions":        ["<emotion>"],       // list, any of: angry, happy, neutral, sad, fearful, disgusted, calm, excited
  "min_duration":    <float|null>,        // seconds
  "max_duration":    <float|null>,        // seconds
  "min_energy":      <float|null>,        // RMS 0-1 (low<0.04, medium~0.06, high>0.09)
  "max_energy":      <float|null>,
  "min_pitch":       <float|null>,        // Hz (low<130, medium~220, high>260)
  "max_pitch":       <float|null>,
  "semantic_query":  "<string>",          // best plain-English description for vector search
  "topic_keywords":  ["<word>"]           // important nouns / verbs from query
}}

User query: "{query}"
"""

def parse_query(user_query: str, llm) -> dict:
    prompt = _PARSE_PROMPT.format(query=user_query)
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        # strip ```json ... ``` fence
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()
    try:
        params = json.loads(raw)
    except Exception:
        params = {}

    # Defaults / normalization
    params.setdefault("emotions",       [])
    params.setdefault("min_duration",   None)
    params.setdefault("max_duration",   None)
    params.setdefault("min_energy",     None)
    params.setdefault("max_energy",     None)
    params.setdefault("min_pitch",      None)
    params.setdefault("max_pitch",      None)
    params.setdefault("semantic_query", user_query)
    params.setdefault("topic_keywords", [])
    if isinstance(params["emotions"], str):
        params["emotions"] = [params["emotions"]] if params["emotions"] else []
    return params


# ──────────────────────────────────────────────
# STEP 2 — HARD FILTERS
# ──────────────────────────────────────────────
def hard_filter(index: list[dict], params: dict) -> list[dict]:
    out = index
    if params["min_duration"] is not None:
        out = [c for c in out if c["duration"] >= params["min_duration"]]
    if params["max_duration"] is not None:
        out = [c for c in out if c["duration"] <= params["max_duration"]]
    if params["min_energy"] is not None:
        out = [c for c in out if c.get("energy", 0) >= params["min_energy"]]
    if params["max_energy"] is not None:
        out = [c for c in out if c.get("energy", 0) <= params["max_energy"]]
    if params["min_pitch"] is not None:
        out = [c for c in out if c.get("pitch", 0) >= params["min_pitch"]]
    if params["max_pitch"] is not None:
        out = [c for c in out if c.get("pitch", 0) <= params["max_pitch"]]
    return out


# ──────────────────────────────────────────────
# STEP 3 — HYBRID SCORING
# ──────────────────────────────────────────────
def _doc_text(entry: dict) -> str:
    """Rich document representation for semantic embedding."""
    scores = entry.get("emotion_scores", {})
    if isinstance(scores, str):
        scores = json.loads(scores)
    emotion_detail = ", ".join(f"{k}: {v:.2f}" for k, v in scores.items())
    keywords = entry.get("transcription", "")
    return (
        f"Transcription: {entry['transcription']}. "
        f"Emotion: {entry['emotion']} ({emotion_detail}). "
        f"Duration: {entry['duration']}s. "
        f"Energy: {entry.get('energy', 0):.4f}. "
        f"Pitch: {entry.get('pitch', 0):.1f}Hz."
    )


def _emotion_score(entry: dict, wanted_emotions: list[str]) -> float:
    """
    Returns the cumulative probability mass of wanted emotions from the
    emotion_scores distribution. Falls back to 0.15 bonus if label matches.
    """
    if not wanted_emotions:
        return 0.0
    scores = entry.get("emotion_scores", {})
    if isinstance(scores, str):
        try:
            scores = json.loads(scores)
        except Exception:
            scores = {}
    total = sum(scores.get(e, 0.0) for e in wanted_emotions)
    if total == 0 and entry.get("emotion") in wanted_emotions:
        total = 0.15
    return total


def hybrid_rank(index: list[dict], params: dict, embedder, top_k: int = 12) -> list[dict]:
    if not index:
        return []

    # Semantic scores
    query_emb = embedder.encode(params["semantic_query"], normalize_embeddings=True)
    docs      = [_doc_text(c) for c in index]
    doc_embs  = embedder.encode(docs, normalize_embeddings=True, batch_size=32)

    scored = []
    for c, emb in zip(index, doc_embs):
        semantic  = float(np.dot(query_emb, emb))          # cosine (normalized)
        emotion   = _emotion_score(c, params["emotions"])   # 0–1 probability mass

        # Keyword overlap bonus
        keywords  = [kw.lower() for kw in params.get("topic_keywords", [])]
        text_low  = c.get("transcription", "").lower()
        kw_hits   = sum(1 for kw in keywords if kw in text_low)
        kw_bonus  = min(kw_hits * 0.03, 0.12)              # cap at 0.12

        # Weighted composite
        composite = 0.55 * semantic + 0.30 * emotion + 0.15 * kw_bonus

        c["_score_semantic"] = round(semantic,  4)
        c["_score_emotion"]  = round(emotion,   4)
        c["_score_kw"]       = round(kw_bonus,  4)
        c["similarity_score"] = round(composite, 4)
        scored.append(c)

    return sorted(scored, key=lambda x: x["similarity_score"], reverse=True)[:top_k]


# ──────────────────────────────────────────────
# STEP 4 — LLM RERANK
# ──────────────────────────────────────────────
def llm_rerank(query: str, candidates: list[dict], llm, top_n: int = 5) -> list[dict]:
    if not candidates:
        return []

    lines = []
    for i, c in enumerate(candidates, 1):
        scores = c.get("emotion_scores", {})
        if isinstance(scores, str):
            try:
                scores = json.loads(scores)
            except Exception:
                scores = {}
        score_str = ", ".join(f"{k}:{v:.2f}" for k, v in scores.items())
        lines.append(
            f'{i}. filename="{c["filename"]}" | emotion={c["emotion"]} ({score_str}) '
            f'| duration={c["duration"]}s | energy={c.get("energy",0):.4f} '
            f'| pitch={c.get("pitch",0):.0f}Hz | semantic_score={c.get("similarity_score",0):.3f}\n'
            f'   Transcription: "{c["transcription"]}"'
        )
    blob = "\n".join(lines)

    prompt = f"""You are an expert audio retrieval system. 
Re-rank the following audio clips for relevance to the user query.

User query: "{query}"

Candidates:
{blob}

Instructions:
- Consider the transcription content, emotion distribution, duration, energy, and pitch.
- Assign a rank (1 = best match).
- Provide a concise reason explaining why the clip fits (or doesn't).

Return ONLY a valid JSON array of objects with keys: rank, filename, reason
"""
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()
    try:
        ranking = json.loads(raw)
    except Exception:
        return candidates[:top_n]

    lookup = {c["filename"]: c for c in candidates}
    final  = []
    for r in sorted(ranking, key=lambda x: x.get("rank", 99)):
        c = lookup.get(r.get("filename"), {})
        if c:
            final.append({**c, "rank": r["rank"], "reason": r.get("reason", "")})
    return final[:top_n]


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def semantic_audio_query(query: str, folder: str, rebuild: bool = False) -> list[dict]:
    if rebuild:
        (Path(folder) / INDEX_FILENAME).unlink(missing_ok=True)

    index    = load_or_build_index(folder)
    llm      = get_llm()
    embedder = get_embedder()

    params     = parse_query(query, llm)
    print(f"\n🔍 Parsed params: {json.dumps(params, indent=2)}")

    filtered   = hard_filter(index, params)
    print(f"📊 After hard filters: {len(filtered)}/{len(index)} candidates")

    if not filtered:
        print("⚠ No candidates after filtering — relaxing filters.")
        filtered = index

    ranked     = hybrid_rank(filtered, params, embedder)
    results    = llm_rerank(query, ranked, llm)
    return results


# ──────────────────────────────────────────────
# PRETTY PRINT
# ──────────────────────────────────────────────
def pretty_print(results: list[dict]):
    if not results:
        print("\n❌ No results found.")
        return
    print("\n" + "="*55)
    print("  TOP RESULTS")
    print("="*55)
    for r in results:
        scores = r.get("emotion_scores", {})
        if isinstance(scores, str):
            try:
                scores = json.loads(scores)
            except Exception:
                scores = {}
        score_str = " | ".join(f"{k}: {v:.2f}" for k, v in scores.items())
        print(f"\n  #{r.get('rank', '?')}  {r['filename']}")
        print(f"  Emotion      : {r['emotion']}  ({score_str})")
        print(f"  Duration     : {r['duration']}s")
        print(f"  Energy       : {r.get('energy', 0):.4f}   Pitch: {r.get('pitch', 0):.1f} Hz")
        print(f"  Hybrid Score : {r.get('similarity_score', 'n/a')}")
        print(f"  Transcription: {r['transcription']}")
        print(f"  Why          : {r.get('reason', '')}")
    print("\n" + "="*55)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic audio retrieval with LLM")
    parser.add_argument("--folder",  required=True,          help="Folder containing audio files")
    parser.add_argument("--query",   default=None,           help="Search query (omit for interactive mode)")
    parser.add_argument("--rebuild", action="store_true",    help="Rebuild index from scratch")
    parser.add_argument("--top",     type=int, default=5,    help="Number of results to return")
    args = parser.parse_args()

    # Pre-load ONCE
    if args.rebuild:
        (Path(args.folder) / INDEX_FILENAME).unlink(missing_ok=True)
    index    = load_or_build_index(args.folder)
    llm      = get_llm()
    embedder = get_embedder()
    print("✅ Models ready.\n")

    def _run(q: str):
        params   = parse_query(q, llm)
        print(f"\n🔍 Parsed: {json.dumps(params, indent=2)}")
        filtered = hard_filter(index, params)
        print(f"📊 After filters: {len(filtered)}/{len(index)}")
        if not filtered:
            print("⚠ No candidates after filtering — relaxing.")
            filtered = index
        ranked  = hybrid_rank(filtered, params, embedder, top_k=max(args.top * 2, 12))
        results = llm_rerank(q, ranked, llm, top_n=args.top)
        pretty_print(results)

    if args.query:
        _run(args.query)
    else:
        print("🎤 Interactive Audio Retrieval Mode  (type 'exit' to quit)\n")
        while True:
            try:
                q = input("Query > ").strip()
                if not q or q.lower() in ("exit", "quit"):
                    break
                _run(q)
                print()
            except (KeyboardInterrupt, EOFError):
                break
        print("\n👋 Goodbye!")