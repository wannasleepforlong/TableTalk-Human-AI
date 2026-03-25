from t4_retreival import semantic_audio_query
results = semantic_audio_query(
    query="happy audio about greetings",
    folder="test"
)
for r in results:
    print(r["rank"], r["filename"], r["reason"])

