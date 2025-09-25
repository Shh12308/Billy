# main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.get("/status")
def status():
    return {"status": "AI running"}

@app.post("/predict")
def predict(data: InputData):
    return {"output": f"Received: {data.text}"}
