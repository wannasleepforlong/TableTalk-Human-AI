"""
t2_tone_classifier.py
─────────────────────
Narrative Tone Classifier for TableTalk recordings.

No labelled training data is needed. Tone is detected via a 3-stage pipeline:

  Stage 1 │ Zero-shot NLI text classifier  (facebook/bart-large-mnli)
           │   → score each tone against the transcription
  Stage 2 │ Acoustic heuristics
           │   → adjust scores using energy, pitch variance, speech rate
  Stage 3 │ LLM verification  (Gemini)
           │   → final label + explanation from all signals combined

Supported tones
───────────────
  • suspense          – building tension, unresolved stakes
  • calm_description  – steady, informative, low-affect narration
  • urgency           – fast-paced, high-stakes, time-pressured
  • dramatic_emphasis – heightened emotion, deliberate pacing, emphasis
  • character_dialogue – distinct voice / persona, conversational

Usage
─────
  python t2_tone_classifier.py --file path/to/audio.wav
  python t2_tone_classifier.py --folder test/               # batch
  python t2_tone_classifier.py --folder test/ --no-llm      # skip LLM (faster)
  python t2_tone_classifier.py --folder test/ --csv out.csv # save results
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import dotenv

dotenv.load_dotenv(override=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TONES = [
    "suspense",
    "calm_description",
    "urgency",
    "dramatic_emphasis",
    "character_dialogue",
]

# Human-readable hypothesis templates fed to NLI model
TONE_HYPOTHESES = {
    "suspense":          "This audio contains tense, suspenseful narration with a sense of danger or mystery.",
    "calm_description":  "This audio contains calm, steady, descriptive narration with no urgency or strong emotion.",
    "urgency":           "This audio conveys urgency, haste, or time pressure with fast-paced speech.",
    "dramatic_emphasis": "This audio has dramatic, emotionally heightened narration with deliberate pauses and emphasis.",
    "character_dialogue":"This audio features a distinct character voice or conversational dialogue performance.",
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# ──────────────────────────────────────────────────────────────────────────────
# LAZY MODEL SINGLETONS
# ──────────────────────────────────────────────────────────────────────────────
_asr       = None
_nli       = None
_llm       = None

def get_asr():
    global _asr
    if _asr is None:
        from transformers import pipeline as hf_pipeline
        print("⏳ Loading Whisper ASR …")
        _asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small")
    return _asr

def get_nli():
    global _nli
    if _nli is None:
        from transformers import pipeline as hf_pipeline
        print("⏳ Loading zero-shot NLI classifier (bart-large-mnli) …")
        _nli = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _nli

def get_llm():
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return _llm

def transcribe(audio_path: str) -> str:
    result = get_asr()(audio_path, generate_kwargs={"language": "en"})
    return result.get("text", "").strip()

def nli_scores(transcription: str) -> dict[str, float]:
    """
    Runs multi-label zero-shot classification and returns a score
    per tone (0–1).  We use multi_label=True so tones are rated
    independently rather than as a mutually-exclusive softmax.
    """
    if not transcription.strip():
        return {t: 0.0 for t in TONES}

    hypotheses = list(TONE_HYPOTHESES.values())
    result = get_nli()(
        transcription,
        candidate_labels=hypotheses,
        multi_label=True,
    )

    # Map hypothesis text → tone name
    hyp_to_tone = {v: k for k, v in TONE_HYPOTHESES.items()}
    scores = {}
    for label, score in zip(result["labels"], result["scores"]):
        tone = hyp_to_tone.get(label)
        if tone:
            scores[tone] = round(float(score), 4)
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ACOUSTIC HEURISTICS
# ──────────────────────────────────────────────────────────────────────────────
def acoustic_features(audio_path: str) -> dict:
    """
    Extracts lightweight acoustic descriptors relevant to narrative tone.
    Returns a dict of normalised signals.
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)
    if len(y) == 0:
        return {}

    duration = librosa.get_duration(y=y, sr=sr)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    mean_energy = float(np.mean(rms))
    std_energy  = float(np.std(rms))          # high std → dynamic range

    # Pitch
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        mean_pitch = float(np.mean(voiced)) if len(voiced) else 0.0
        std_pitch  = float(np.std(voiced))  if len(voiced) else 0.0
    except Exception:
        mean_pitch, std_pitch = 0.0, 0.0

    # Speech rate proxy: zero-crossing rate (fast speech → higher ZCR)
    zcr        = librosa.feature.zero_crossing_rate(y)[0]
    mean_zcr   = float(np.mean(zcr))

    # Tempo (BPM proxy via onset strength)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo     = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])
    except Exception:
        tempo = 0.0

    return {
        "duration":    round(duration,    2),
        "mean_energy": round(mean_energy, 5),
        "std_energy":  round(std_energy,  5),
        "mean_pitch":  round(mean_pitch,  2),
        "std_pitch":   round(std_pitch,   2),
        "mean_zcr":    round(mean_zcr,    5),
        "tempo":       round(tempo,       2),
    }


def acoustic_adjustments(scores: dict[str, float], acf: dict) -> dict[str, float]:
    """
    Applies small heuristic boosts / penalties to the NLI scores based on
    acoustic signals.  All adjustments are small (±0.05 to ±0.15) to avoid
    overriding the NLI signal.

    Heuristic rationale
    ───────────────────
    suspense          → low sustained energy, wide pitch range, slow tempo
    calm_description  → low energy variance, moderate pitch, moderate ZCR
    urgency           → high energy, high ZCR, fast tempo
    dramatic_emphasis → high energy variance (dynamic), wide pitch swings
    character_dialogue→ high pitch variance (character voices), moderate rate
    """
    adj = dict(scores)
    if not acf:
        return adj

    e      = acf.get("mean_energy", 0.06)
    std_e  = acf.get("std_energy",  0.01)
    p      = acf.get("mean_pitch",  200)
    std_p  = acf.get("std_pitch",   30)
    zcr    = acf.get("mean_zcr",    0.05)
    tempo  = acf.get("tempo",       100)

    # suspense: quiet but dynamic (building tension)
    if e < 0.055 and std_e > 0.008:
        adj["suspense"] = min(1.0, adj.get("suspense", 0) + 0.10)
    if tempo > 130:                                              # fast rate → less suspense
        adj["suspense"] = max(0.0, adj.get("suspense", 0) - 0.05)

    # calm_description: low variance in everything
    if std_e < 0.006 and std_p < 25 and zcr < 0.06:
        adj["calm_description"] = min(1.0, adj.get("calm_description", 0) + 0.12)

    # urgency: loud + fast
    if e > 0.07 and (zcr > 0.07 or tempo > 130):
        adj["urgency"] = min(1.0, adj.get("urgency", 0) + 0.12)

    # dramatic_emphasis: high energy swing, wide pitch
    if std_e > 0.012 and std_p > 40:
        adj["dramatic_emphasis"] = min(1.0, adj.get("dramatic_emphasis", 0) + 0.10)
    if e > 0.08:
        adj["dramatic_emphasis"] = min(1.0, adj.get("dramatic_emphasis", 0) + 0.05)

    # character_dialogue: high pitch variance (distinct voices / inflections)
    if std_p > 50:
        adj["character_dialogue"] = min(1.0, adj.get("character_dialogue", 0) + 0.12)

    return {k: round(v, 4) for k, v in adj.items()}


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — LLM VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────
_LLM_PROMPT = """You are an expert in narrative audio and voice performance.
Classify the dominant narrative tone of the following audio clip.

Possible tones:
  • suspense          – building tension, unresolved stakes, hushed dread
  • calm_description  – steady, informative narration, low affect
  • urgency           – fast-paced, high-stakes, time-pressured delivery
  • dramatic_emphasis – emotionally heightened, deliberate pacing, strong emphasis
  • character_dialogue – distinct character voice / accent, conversational persona

Evidence:
  Transcription : "{transcription}"
  Duration      : {duration}s
  Mean energy   : {mean_energy}  (low<0.04 │ med~0.06 │ high>0.09)
  Energy std    : {std_energy}   (high std → dynamic, dramatic)
  Mean pitch    : {mean_pitch} Hz
  Pitch std     : {std_pitch} Hz (high std → varied intonation)
  Speech tempo  : {tempo} BPM
  NLI scores    : {nli_scores}

Respond with ONLY valid JSON:
{{
  "tone":        "<one of the five tones above>",
  "confidence":  <float 0-1>,
  "reason":      "<one concise sentence explaining the choice>"
}}"""


def llm_verify(transcription: str, acf: dict, scores: dict, llm) -> dict:
    prompt = _LLM_PROMPT.format(
        transcription=transcription[:800],          # truncate very long texts
        duration=acf.get("duration",    "?"),
        mean_energy=acf.get("mean_energy", "?"),
        std_energy=acf.get("std_energy",   "?"),
        mean_pitch=acf.get("mean_pitch",   "?"),
        std_pitch=acf.get("std_pitch",     "?"),
        tempo=acf.get("tempo",             "?"),
        nli_scores=json.dumps(scores, indent=2),
    )
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        # Fallback: pick highest NLI score
        best = max(scores, key=scores.get)
        return {"tone": best, "confidence": scores[best], "reason": "LLM parse failed; using NLI argmax."}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def classify_tone(audio_path: str, use_llm: bool = True) -> dict:
    """
    Full 3-stage pipeline for a single audio file.

    Returns
    ───────
    {
      "file":         str,
      "transcription": str,
      "acf":          dict,           # acoustic features
      "nli_scores":   dict,           # per-tone NLI scores (post-heuristic)
      "tone":         str,            # final predicted tone
      "confidence":   float,
      "reason":       str,
    }
    """
    print(f"\n  🎙  {os.path.basename(audio_path)}")

    # Stage 1a — transcribe
    print("    ↳ Transcribing …")
    transcription = transcribe(audio_path)
    print(f"    ↳ Text: {transcription[:120]}{'…' if len(transcription)>120 else ''}")

    # Stage 1b — NLI
    print("    ↳ Zero-shot NLI scoring …")
    raw_scores = nli_scores(transcription)

    # Stage 2 — acoustic
    print("    ↳ Computing acoustic features …")
    acf = acoustic_features(audio_path)
    adj_scores = acoustic_adjustments(raw_scores, acf)

    # Stage 3 — LLM
    if use_llm:
        print("    ↳ LLM verification …")
        llm = get_llm()
        verdict = llm_verify(transcription, acf, adj_scores, llm)
    else:
        best = max(adj_scores, key=adj_scores.get)
        verdict = {"tone": best, "confidence": adj_scores[best], "reason": "LLM skipped."}

    return {
        "file":          audio_path,
        "filename":      os.path.basename(audio_path),
        "transcription": transcription,
        "acf":           acf,
        "nli_scores":    adj_scores,
        "tone":          verdict["tone"],
        "confidence":    round(float(verdict.get("confidence", 0)), 3),
        "reason":        verdict.get("reason", ""),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Narrative tone classifier for TableTalk audio")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",   help="Single audio file to classify")
    group.add_argument("--folder", help="Folder of audio files (batch mode)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM stage (faster, less accurate)")
    parser.add_argument("--csv",    default=None,         help="Save results to this CSV path")
    args = parser.parse_args()

    use_llm = not args.no_llm

    # Pre-load models
    get_asr()
    get_nli()
    if use_llm:
        get_llm()
    print("\n✅ Models ready.\n")

    if args.file:
        files = [args.file]
    else:
        folder = args.folder
        files  = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ]
        print(f"📂 Found {len(files)} audio file(s) in '{folder}'\n")

    rows = []
    for f in files:
        try:
            result = classify_tone(f, use_llm=use_llm)
            rows.append({
                "filename":      result["filename"],
                "tone":          result["tone"],
                "confidence":    result["confidence"],
                "reason":        result["reason"],
                "transcription": result["transcription"],
                **{f"nli_{k}": v for k, v in result["nli_scores"].items()},
                **result["acf"],
            })
        except Exception as e:
            print(f"  ⚠ Error processing {f}: {e}")

    if args.csv and rows:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\n✅ Results saved → {args.csv}")

    if rows:
        print(f"\n📊 Summary  ({len(rows)} file(s))")
        tone_counts = pd.Series([r["tone"] for r in rows]).value_counts()
        for tone, count in tone_counts.items():
            print(f"   {tone:<22s} {count}")
