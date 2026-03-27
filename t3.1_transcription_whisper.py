import os
import re
from transformers import pipeline
from jiwer import wer, cer
from tqdm import tqdm

TEST_FOLDER = "test"
GT_PATH = os.path.join(TEST_FOLDER, "ground_truth.txt")
MODEL_ID = "openai/whisper-small"


asr = pipeline("automatic-speech-recognition", model=MODEL_ID)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
gt_dict = {}
with open(GT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            continue
        file_id, text = parts
        gt_dict[file_id] = text

print(f"Loaded {len(gt_dict)} ground truth entries")

refs = []
hyps = []

for file_id, ref_text in tqdm(gt_dict.items()):
    audio_file = f"{file_id}.flac"   # change to .mp3 if needed
    audio_path = os.path.join(TEST_FOLDER, audio_file)

    if not os.path.exists(audio_path):
        print(f"❌ Missing: {audio_file}")
        continue

    try:
        result = asr(audio_path, generate_kwargs={"language": "en"})
        hyp_text = result["text"]

        ref = clean_text(ref_text)
        hyp = clean_text(hyp_text)

        refs.append(ref)
        hyps.append(hyp)

    except Exception as e:
        print(f"⚠ Skipped {audio_file}: {e}")


if refs:
    wer_score = wer(refs, hyps)
    cer_score = cer(refs, hyps)

    print(f"\nWER: {wer_score:.4f}")
    print(f"CER: {cer_score:.4f}")
else:
    print("❌ No valid samples processed.")


# WER: 0.0370
# CER: 0.0234