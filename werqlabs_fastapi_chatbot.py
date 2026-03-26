# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import ask_bot

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
def chat(req: ChatRequest):
    response = ask_bot(req.message, req.session_id)
    return {"response": response}