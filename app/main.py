from fastapi import FastAPI

app = FastAPI(title="IMDb Sentiment API")

@app.get("/health")
def health_check():
    return {"status": "ok"}
