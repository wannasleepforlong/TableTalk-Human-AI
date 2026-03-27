import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
import librosa
import dotenv

dotenv.load_dotenv(override=True)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
INDEX_FILENAME   = "audio_index.csv"
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
TONES = [
    "character_dialogue", 
    "suspense", 
    "dramatic_emphasis", 
    "urgency", 
    "calm_description"
]
RETRIEVAL_PARAMS = {
    "weights": {
        "text": 0.5,
        "tone": 0.3,
        "numeric": 0.2
    },
    "thresholds": {
        "duration": {"low": 5.0,  "high": 15.0},
        "energy":   {"low": 0.0, "high": 0.03},
        "pitch":    {"low": 130.0, "high": 250.0}
    }
}

_embedder = None
_llm      = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("⏳ Loading sentence embedder …")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

class TinyLlamaLLM:
    """Simple wrapper to mimic LangChain's invoke interface for local TinyLlama."""
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"⏳ Loading local LLM: {model_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.pipeline = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )

    def invoke(self, prompt: str):
        # Format for TinyLlama Chat
        formatted_prompt = f"<|system|>\nYou are a helpful audio retrieval assistant that outputs ONLY JSON.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        outputs = self.pipeline(formatted_prompt)
        text = outputs[0]["generated_text"]
        # Extract assistant response
        response = text.split("<|assistant|>")[-1].strip().replace("</s>", "")
        
        # Simple class to mimic LangChain response object
        class Response:
            def __init__(self, content): self.content = content
        return Response(response)

class MistralLLM:
    """Wrapper for Mistral AI API."""
    def __init__(self, api_key, model="mistral-small-latest"):
        print(f"⏳ Connecting to Mistral AI: {model} …")
        self.llm = ChatMistralAI(model=model, api_key=api_key)

    def invoke(self, prompt: str):
        # Mistral uses standard Chat message format
        response = self.llm.invoke(prompt)
        return response

def get_llm(model_type="local", model_id=None):
    global _llm
    if _llm is not None:
        return _llm
        
    if model_type == "mistral":
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY not found in environment.")
        _llm = MistralLLM(MISTRAL_API_KEY, model="mistral-small-latest")
    else:
        # Default to local TinyLlama
        _llm = TinyLlamaLLM(model_id=model_id or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return _llm

def load_index(csv_path: str = "outputs/bart_output.csv") -> list[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing feature file: {csv_path}")
    print(f"✅ Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


# ──────────────────────────────────────────────
# STEP 1 — LLM QUERY PARSING
# ──────────────────────────────────────────────
_PARSE_PROMPT = """You are an audio retrieval assistant. Parse the user query into structured JSON requirements.

Return ONLY valid JSON with these exact keys:
{{
  "semantic_query":  "<string>",     // Optimized English description for vector search on transcription
  "tone_vector": {{                   // Score each tone from 0.0 to 1.0 based on user's desire
    "character_dialogue": <float>,
    "suspense": <float>,
    "dramatic_emphasis": <float>,
    "urgency": <float>,
    "calm_description": <float>
  }},
  "duration_level":  "<High|Medium|Low|None>",
  "energy_level":    "<High|Medium|Low|None>",
  "pitch_level":     "<High|Medium|Low|None>"
}}

User query: "{query}"
"""

def parse_query(user_query: str, llm) -> dict:
    prompt = _PARSE_PROMPT.format(query=user_query)
    raw = llm.invoke(prompt).content.strip()
    # Clean possible markdown fence
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()
    try:
        params = json.loads(raw)
        print(params)
    except Exception:
        print(f"⚠ Failed to parse LLM JSON: {raw}")
        params = {}

    # Normalization & Defaults
    params.setdefault("semantic_query", user_query)
    params.setdefault("tone_vector", {t: 0.0 for t in TONES})
    params.setdefault("duration_level", "None")
    params.setdefault("energy_level", "None")
    params.setdefault("pitch_level", "None")
    
    return params


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm > 0 else 0.0

def hybrid_rank(index: list[dict], params: dict, embedder, top_k: int = 12) -> list[dict]:
    if not index:
        return []

    # 1. Semantic Transcription Score
    query_emb = embedder.encode(params["semantic_query"], normalize_embeddings=True)
    transcriptions = [str(c.get("transcription", "")) for c in index]
    doc_embs = embedder.encode(transcriptions, normalize_embeddings=True, batch_size=32)

    # 2. Tone Vector Score
    q_tone_vec = np.array([params["tone_vector"].get(t, 0.0) for t in TONES])
    
    scored = []
    for c, t_emb in zip(index, doc_embs):
        # Text similarity
        text_sim = float(np.dot(query_emb, t_emb))

        # Tone similarity
        c_tone_vec = np.array([c.get(f"nli_{t}", 0.0) for t in TONES])
        tone_sim = cosine_similarity(q_tone_vec, c_tone_vec)

        # Numeric matches (level-based scoring)
        num_score = 0.0
        active_features = 0
        thresh = RETRIEVAL_PARAMS["thresholds"]

        for feat in ["duration", "energy", "pitch"]:
            level = params.get(f"{feat}_level", "None")
            if level != "None":
                val = c.get(feat if feat == "duration" else f"mean_{feat}", 0.0)
                low, high = thresh[feat]["low"], thresh[feat]["high"]
                
                feat_score = 0.0
                if level == "High" and val > high:
                    feat_score = 1.0
                elif level == "Medium" and low <= val <= high:
                    feat_score = 1.0
                elif level == "Low" and val < low:
                    feat_score = 1.0
                
                num_score += feat_score
                active_features += 1
        
        num_match = num_score / active_features if active_features > 0 else 0.0

        # Weighted final score
        # Text: 50%, Tone: 30%, Num: 20%
        w = RETRIEVAL_PARAMS["weights"]
        final_score = (w["text"] * text_sim) + (w["tone"] * tone_sim) + (w["numeric"] * num_match)

        c["similarity_score"] = float(round(final_score, 4))
        c["_score_text"]      = float(round(text_sim, 4))
        c["_score_tone"]      = float(round(tone_sim, 4))
        c["_score_num"]       = float(round(num_match, 4))
        scored.append(c)

    return sorted(scored, key=lambda x: x["similarity_score"], reverse=True)[:top_k]


def semantic_audio_query(query: str, top_n: int =3) -> list[dict]:
    index    = load_index()
    llm      = get_llm()
    embedder = get_embedder()

    params   = parse_query(query, llm)
    print(f"\n🔍 Parsed: {json.dumps(params, indent=2)}")

    results  = hybrid_rank(index, params, embedder, top_k=top_n)
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
    for i, r in enumerate(results, 1):
        tone_str = " | ".join(f"{t}: {r.get(f'nli_{t}',0):.2f}" for t in TONES)
        print(f"\n  #{i}  {r['filename']}")
        print(f"  Similarity   : {r.get('similarity_score', 'n/a')}")
        print(f"  Transcription: {r.get('transcription','')}")
    print("\n" + "="*55)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic audio retrieval with LLM")
    parser.add_argument("--query",   default=None,           help="Search query (omit for interactive mode)")
    parser.add_argument("--top",     type=int, default=5,    help="Number of results to return")
    parser.add_argument("--model",   choices=["local", "mistral"], default="local", help="LLM backend to use")
    args = parser.parse_args()

    if args.query:
        # Pre-load for CLI
        index    = load_index()
        llm      = get_llm(model_type=args.model)
        embedder = get_embedder()
        
        params   = parse_query(args.query, llm)
        results  = hybrid_rank(index, params, embedder, top_k=args.top)
        pretty_print(results)
    else:
        # Interactive
        index    = load_index()
        llm      = get_llm(model_type=args.model)
        embedder = get_embedder()
        print(f"✅ Models & Data ready (Backend: {args.model}).\n")
        
        print("🎤 Interactive Audio Retrieval Mode  (type 'exit' to quit)\n")
        while True:
            try:
                q = input("Query > ").strip()
                if not q or q.lower() in ("exit", "quit"):
                    break
                
                params  = parse_query(q, llm)
                print(f"🔍 Parsed: {json.dumps(params, indent=2)}")
                results = hybrid_rank(index, params, embedder, top_k=args.top)
                pretty_print(results)
            except Exception as e:
                print(f"❌ Error: {e}")
                