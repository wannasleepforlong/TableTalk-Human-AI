import os
import pandas as pd
from transformers import pipeline
from jiwer import wer, cer
from tqdm import tqdm
import re

TEST_FOLDER = "test"
TSV_PATH = os.path.join(TEST_FOLDER, "ss-corpus-en.tsv")
MODEL_ID = "openai/whisper-small"

asr = pipeline("automatic-speech-recognition", model=MODEL_ID)
df = pd.read_csv(TSV_PATH, sep="\t")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()    
    return text

refs = []
hyps = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    audio_file = row["audio_file"]
    ref_text = row["transcription"]
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
    print(f"WER: {wer_score:.4f}")
    print(f"CER: {cer_score:.4f}")
else:
    print("No valid samples processed.")


# WER: 0.0926
# CER: 0.0374
