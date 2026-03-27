import pandas as pd
import os
import re
from tqdm import tqdm
from jiwer import wer, cer

CSV_PATH = "outputs/bart_output.csv"
MERGED_CSV = "outputs/bart_output_merged.csv"
TEST_FOLDER = "test"
GT_PATH = os.path.join(TEST_FOLDER, "ground_truth.txt")

df = pd.read_csv(CSV_PATH, usecols=["filename", "transcription"])

def get_base(fname):
    base = os.path.splitext(fname)[0]
    if "_chunk_" in base:
        base = base.split("_chunk_")[0]
    return base

def get_chunk_index(fname):
    if "_chunk_" in fname:
        return int(fname.split("_chunk_")[1].split(".")[0])
    return 0

df["base"] = df["filename"].apply(get_base)
df["chunk_id"] = df["filename"].apply(get_chunk_index)
df = df.sort_values(by=["base", "chunk_id"])

def merge_texts(texts):
    merged = texts[0]
    for next_text in texts[1:]:
        merged_words = merged.split()
        next_words = next_text.split()
        max_overlap = min(len(merged_words), len(next_words))
        overlap_size = 0
        for i in range(max_overlap, 0, -1):
            if merged_words[-i:] == next_words[:i]:
                overlap_size = i
                break
        merged = merged + " " + " ".join(next_words[overlap_size:])
    return merged

merged_df = (
    df.groupby("base")["transcription"]
    .apply(lambda x: merge_texts(list(x)))
    .reset_index()
)

merged_df.rename(columns={"base": "filename"}, inplace=True)
merged_df.to_csv(MERGED_CSV, index=False)

pred_dict = dict(zip(merged_df["filename"], merged_df["transcription"]))

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

refs = []
hyps = []

for file_id, ref_text in tqdm(gt_dict.items()):
    if file_id not in pred_dict:
        continue
    hyp_text = pred_dict[file_id]
    refs.append(clean_text(ref_text))
    hyps.append(clean_text(hyp_text))

if refs:
    print(f"WER: {wer(refs, hyps):.4f}")
    print(f"CER: {cer(refs, hyps):.4f}")


# WER: 0.0487
# CER: 0.0345