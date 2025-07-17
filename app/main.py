from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from app.analyzer import analyze_sentiment

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join("app", "static", "index.html"))

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze(input: TextInput):
    return analyze_sentiment(input.text)
