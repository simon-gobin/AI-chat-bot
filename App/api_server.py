from fastapi import FastAPI, Query
from pydantic import BaseModel
from app import RoleplayAssistant

assistant = RoleplayAssistant("story_state.json")
assistant.login_to_huggingface()
assistant.load_models()
assistant.json_file()
assistant.init_story()

app = FastAPI()

class MessageRequest(BaseModel):
    message: str

@app.get("/chat")
def get_response(message: str = Query(...)):
    assistant.chat_add("user", message)
    reply = assistant.model_output([
        {"role": "system", "content": "Start roleplay."},
        {"role": "user", "content": message}
    ])
    assistant.chat_add("system", reply)
    return {"reply": reply}
