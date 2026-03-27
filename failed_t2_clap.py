"""
t2_tone_classifier.py
─────────────────────
Narrative Tone Classifier for TableTalk recordings.

Uses CLAP (Contrastive Language-Audio Pretraining) for zero-shot audio-text
matching — no transcription or acoustic heuristics needed.

CLAP encodes both the audio and each tone description into a shared embedding
space, then picks the tone whose text prompt is most similar to the audio.

Supported tones
───────────────
  • suspense          – building tension, unresolved stakes
  • calm              – steady, informative, low-affect narration
  • urgency           – fast-paced, high-stakes, time-pressured
  • dramatic_emphasis – heightened emotion, deliberate pacing, emphasis
"""

import os
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import ClapModel, ClapProcessor

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TONES = [
    "suspense",
    "calm",
    "urgency",
    "dramatic_emphasis",
]

TONE_PROMPTS = {
    "suspense":          "tense suspenseful narration with hushed voice and a sense of danger or mystery",
    "calm":              "calm steady narration with no urgency, flat affect, and even pacing",
    "urgency":           "urgent fast-paced speech conveying high stakes and time pressure",
    "dramatic_emphasis": "dramatically heightened narration with strong emotional emphasis and deliberate pauses",
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
CLAP_MODEL_ID    = "laion/clap-htsat-unfused"    
SAMPLE_RATE      = 48_000                        

_model     = None
_processor = None

def get_clap():
    global _model, _processor
    if _model is None:
        print("⏳ Loading CLAP model …")
        _processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        _model     = ClapModel.from_pretrained(CLAP_MODEL_ID)
        _model.eval()
    return _model, _processor


def load_audio(audio_path: str) -> np.ndarray:
    """Load and resample audio to CLAP's expected 48 kHz mono."""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y


def clap_scores(audio: np.ndarray) -> dict[str, float]:
    """
    Computes cosine similarity between the audio embedding and each tone
    prompt embedding.  Returns a softmax-normalised score dict.
    """
    model, processor = get_clap()

    prompts = list(TONE_PROMPTS.values())

    inputs = processor(
        text=prompts,
        audios=audio,
        return_tensors="pt",
        padding=True,
        sampling_rate=SAMPLE_RATE,
    )

    with torch.no_grad():
        audio_embed = model.get_audio_features(
            input_features=inputs["input_features"]
        )                                              # (1, D)
        text_embed = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )                                              # (n_tones, D)

    # Normalise
    audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
    text_embed  = text_embed  / text_embed.norm(dim=-1, keepdim=True)

    # Cosine similarities  →  softmax  →  probabilities
    logits = (audio_embed @ text_embed.T).squeeze(0)   # (n_tones,)
    probs  = torch.softmax(logits * 100, dim=0).numpy() # temperature=100

    return {tone: round(float(p), 4) for tone, p in zip(TONES, probs)}


def classify_tone(audio_path: str) -> dict:
    """
    Classify the narrative tone of a single audio file via CLAP.

    Returns
    ───────
    {
      "filename":   str,
      "tone":       str,    # predicted tone label
      "confidence": float,  # softmax probability of top tone
      "scores":     dict,   # per-tone softmax probabilities
    }
    """
    print(f"  🎙  {os.path.basename(audio_path)}")
    audio  = load_audio(audio_path)
    scores = clap_scores(audio)
    best   = max(scores, key=scores.get)
    return {
        "filename":   os.path.basename(audio_path),
        "tone":       best,
        "confidence": scores[best],
        "scores":     scores,
    }


# ──────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FOLDER     = "test_processed"
    OUTPUT_CSV = "outputs/clap_output.csv"

    get_clap()
    print("\n✅ Model ready.\n")

    files = [
        os.path.join(FOLDER, f)
        for f in sorted(os.listdir(FOLDER))
        if os.path.isfile(os.path.join(FOLDER, f))
        and os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]
    print(f"📂 Found {len(files)} audio file(s) in '{FOLDER}'\n")

    rows = []
    for path in files:
        try:
            result = classify_tone(path)
            rows.append({
                "filename":   result["filename"],
                "tone":       result["tone"],
                "confidence": result["confidence"],
                **{f"score_{k}": v for k, v in result["scores"].items()},
            })
        except Exception as e:
            print(f"  ⚠ Error processing {path}: {e}")

    if rows:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Results saved → {OUTPUT_CSV}")

        print(f"\n📊 Summary ({len(rows)} file(s))")
        for tone, count in df["tone"].value_counts().items():
            print(f"   {tone:<22s} {count}")