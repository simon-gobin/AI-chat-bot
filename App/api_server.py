from fastapi import FastAPI
from pydantic import BaseModel
from app.roleplay_assistant import RoleplayAssistant
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("🚀 API server is starting...")


# Initialize the assistant and models
assistant = RoleplayAssistant(json_path="story_state.json")
assistant.login_to_huggingface()
assistant.load_models()
assistant.json_file()

app = FastAPI()

class ChatMessage(BaseModel):
    message: str

class LanguageUpdate(BaseModel):
    telegram_id: int
    lang_code: str

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


@app.post("/language")
def set_language(payload: LanguageUpdate):
    user_id = payload.telegram_id
    lang = payload.lang_code

    if user_id not in assistant.story_state:
        assistant.story_state[user_id] = {}

    assistant.story_state[user_id]["Language"] = lang
    assistant.save_json()
    return {"message": f"Language set to {lang} for user {user_id}"}
