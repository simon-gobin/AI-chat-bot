from fastapi import FastAPI
from pydantic import BaseModel
from app.roleplay_assistant import RoleplayAssistant
import os

# Initialize the assistant and models
assistant = RoleplayAssistant(json_path="story_state.json")
assistant.login_to_huggingface()
assistant.load_models()
assistant.json_file()

app = FastAPI()

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(payload: ChatMessage):
    reply = assistant.respond(payload.message)
    return {"reply": reply}

@app.get("/status")
def get_status():
    return assistant.story_state

@app.post("/reset")
def reset_story():
    assistant.story_state = {
        'Language': None,
        'System Character': None,
        'User Character': None,
        'Situation': None,
        'chat': [],
        'Summary of the situation': None,
    }
    assistant.save_json()
    return {"message": "Story state has been reset."}

@app.get("/languages")
def list_languages():
    from app.roleplay_assistant import LANGUAGES
    return {k: {"name": v[0], "code": v[1]} for k, v in LANGUAGES.items()}
