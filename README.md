# TableTalk Human-AI Technical Test

This repository contains my submission for the TableTalk Human-AI Technical Test. The goal of this project is to build a prototype system that organizes and retrieves narrative voice recordings used in interactive storytelling environments.

The project is divided into four main tasks as outlined in the assignment, plus an exploration of the bonus storytelling analysis features.

## Requirements

1. **Python 3.11** (Recommended)
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file (if you want to test the **optional** Gemini / Mistral API pathways):
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   ```
4. Place your raw audio dataset inside a root folder named `test/`. Supported formats are `.wav`, `.mp3`, `.flac`, etc.

---

## Task 1: Audio Processing Pipeline
**File:** `t1_feature_extraction.py`

This script processes the raw audio dataset into structured features tailored for machine learning.

**Features Extracted:**
- **Normalization:** Audio levels are RMS normalized.
- **Trimming:** Silence is trimmed from the beginning and end of clips.
- **Segmentation:** Audio is chopped into 15-second overlapping chunks to cap sequence length for processing.
- **Acoustic Features:** 
  - MFCCs + Delta MFCCs (mean & std)
  - Chroma (mean & std)
  - Spectral descriptors (centroid, bandwidth, rolloff, contrast)
  - Zero-Crossing Rate (ZCR) and RMS Energy
  - Pitch (F0 via YIN algorithm)
  - Tonnetz (harmonic features)

**Usage:**
```bash
python t1_feature_extraction.py
```
**Output:** 
- Processed, chunked audio files saved in `test_processed/`
- Standardized feature dataset saved to `outputs/features_scaled.csv`
- A fitted Standard Scaler saved to `outputs/scaler.pkl`

---

## Task 2: Narrative Tone Classification
**Files:** `t2.1_tone_classifier.py` and `t2.2_model_training.py`

Since the dataset lacked manually-labelled emotional tones for storytelling, I used a Weak Supervision approach to pseudo-label the data using a text-based zero-shot Natural Language Inference (NLI) model (`facebook/bart-large-mnli`), heavily guided by acoustic heuristics extracted from the audio (such as pitch standard deviation or energy). 

I also tried using CLAP which could be run using `failed_t2_clap.py` but it provided poor results, as could be seen by  outputs\clap_output.csv.

The target classes are: `suspense`, `calm`, `urgency`, and `dramatic_emphasis`.

Once pseudo-labelled in `t2.1`, we train an Artificial Neural Network (ANN) in `t2.2` to classify these emotional tones directly from the raw acoustic features (MFCCs, energy, pitch) extracted in Task 1.

**Usage:**
```bash
# Generate pseudo-labels
python t2.1_tone_classifier.py

# Train the NN
python t2.2_model_training.py
```
**Output:** 
- A trained Keras neural network model: `outputs/emotion_model.keras`
- Performance evaluation metrics printed to the console (Accuracy, F1, Confusion Matrix).

---

## Task 3: AI-Based Transcription
**File:** `t3.1_transcription_whisper.py`

This script provides an automatic speech recognition (ASR) pipeline to transcribe the narrative audio recordings utilizing the `openai/whisper-tiny` model (can be swapped for `-small` or `-base`). 

It calculates accuracy metrics (Word Error Rate (WER) and Character Error Rate (CER)) against a ground truth file using the `jiwer` library. Pre-processing steps remove punctuation and parse tags to accurately evaluate the transcripts.

**Usage:**
```bash
python t3.1_transcription_whisper.py
```

---

## Task 4: Narrative Audio Retrieval (Prototype)
**File:** `t4_retreival.py`

This is the interactive retrieval tool demonstrating the ability to query narrative descriptors using natural language. It combines **Dense Retrieval (Semantic Search)** with **Numeric/Acoustic Filters**.

1. An LLM parses the user text into JSON criteria (desired tone, pitch, energy levels, semantic text match).
2. The transcriptions are encoded into Vector Embeddings (`all-MiniLM-L6-v2`) for semantic search.
3. A Hybrid Scoring algorithm scores each audio clip by calculating the distance between the user's semantic criteria and the clip's actual acoustic profiles.

**Example Queries Supported:**
- *"Soothing short audio about water."*
- *"Short audio mentions the date August 17 or a nearby date."*
- *"Dramatic audio about sea."*

**Usage:**
```bash
python t4_retreival.py
```
