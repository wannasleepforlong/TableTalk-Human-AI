from transformers import pipeline
model_id = "openai/whisper-small"
asr = pipeline("automatic-speech-recognition",model=model_id,)
audio_file = "CREMA-D/1001_IEO_ANG_HI.wav"
result = asr(audio_file, generate_kwargs={"language": "en"})

print("Transcription:", result["text"])