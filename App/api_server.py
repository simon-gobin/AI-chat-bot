from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
import traceback
from typing import Dict, Optional
from contextlib import asynccontextmanager
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: str


class LanguageUpdate(BaseModel):
    user_id: str
    lang_code: str


class ImageRequest(BaseModel):
    user_id: str
    prompt: str


class ApiResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    message: Optional[str] = None
    error: Optional[str] = None


# Multi-user session manager
class MultiUserAssistantManager:
    def __init__(self):
        self.user_sessions: Dict[str, 'RoleplayAssistant'] = {}
        self.model_manager = None
        self.max_concurrent_users = 10

    async def load_shared_models(self):
        """Load models once for all users"""
        from app import RoleplayAssistant
        logger.info("🔄 Loading shared AI models...")

        # Create a single instance for model loading
        self.model_template = RoleplayAssistant("template.json")
        await asyncio.to_thread(self.model_template.login_to_huggingface)
        await asyncio.to_thread(self.model_template.load_models)
        logger.info("✅ Models loaded successfully")

    def get_user_session(self, user_id: str) -> 'RoleplayAssistant':
        """Get or create user session"""
        if len(self.user_sessions) >= self.max_concurrent_users:
            # Clean up inactive sessions
            self._cleanup_inactive_sessions()

        if user_id not in self.user_sessions:
            from app import RoleplayAssistant

            # Create user-specific session
            user_json_path = f"user_states/user_{user_id}.json"
            os.makedirs("user_states", exist_ok=True)

            session = RoleplayAssistant(user_json_path)
            # Share loaded models instead of reloading
            if self.model_template:
                session.model = self.model_template.model
                session.tokenizer = self.model_template.tokenizer
                session.llm = self.model_template.llm
                session.pipe = self.model_template.pipe
                session.pipe_sum = self.model_template.pipe_sum
                session.translation_model = self.model_template.translation_model
                session.translation_tokenizer = self.model_template.translation_tokenizer

            session.json_file()
            self.user_sessions[user_id] = session
            logger.info(f"👤 Created session for user {user_id}")

        return self.user_sessions[user_id]

    def _cleanup_inactive_sessions(self):
        """Remove sessions that haven't been used recently"""
        # Implementation for cleanup based on last activity
        pass

    async def cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("🧹 Cleaning up user sessions...")
        for session in self.user_sessions.values():
            # Save any pending state
            try:
                session.save_json()
            except Exception as e:
                logger.error(f"Error saving session: {e}")


# Global manager instance
manager = MultiUserAssistantManager()


# FastAPI lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 API server starting...")
    try:
        await manager.load_shared_models()
        logger.info("✅ Startup complete")
        yield
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 API server shutting down...")
        await manager.cleanup()
        logger.info("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Roleplay AI Assistant API",
    description="Multi-user AI roleplay assistant with image generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Routes
@app.post("/chat", response_model=ApiResponse)
async def chat_endpoint(payload: ChatMessage):
    """Handle chat messages for a specific user"""
    try:
        session = manager.get_user_session(payload.user_id)

        # Process message asynchronously
        reply = await asyncio.to_thread(session.respond, payload.message)

        return ApiResponse(
            success=True,
            data={"reply": reply},
            message="Message processed successfully"
        )

    except Exception as e:
        logger.error(f"Chat error for user {payload.user_id}: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.get("/status/{user_id}", response_model=ApiResponse)
async def get_user_status(user_id: str):
    """Get current story state for a user"""
    try:
        session = manager.get_user_session(user_id)
        return ApiResponse(
            success=True,
            data=session.story_state,
            message="Status retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Status error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", response_model=ApiResponse)
async def reset_user_story(payload: ChatMessage):
    """Reset story state for a specific user"""
    try:
        session = manager.get_user_session(payload.user_id)
        session.story_state = {
            'Language': None,
            'System Character': None,
            'User Character': None,
            'Situation': None,
            'chat': [],
            'Summary of the situation': None,
        }
        session.save_json()

        return ApiResponse(
            success=True,
            message=f"Story state reset for user {payload.user_id}"
        )
    except Exception as e:
        logger.error(f"Reset error for user {payload.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/language", response_model=ApiResponse)
async def set_language(payload: LanguageUpdate):
    """Set language preference for a user"""
    try:
        session = manager.get_user_session(payload.user_id)
        session.story_state['Language'] = payload.lang_code
        session.save_json()

        return ApiResponse(
            success=True,
            message=f"Language set to {payload.lang_code} for user {payload.user_id}"
        )
    except Exception as e:
        logger.error(f"Language error for user {payload.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-image", response_model=ApiResponse)
async def generate_image_endpoint(payload: ImageRequest, background_tasks: BackgroundTasks):
    """Generate image for user (async background task)"""
    try:
        # Add to background tasks
        background_tasks.add_task(
            generate_image_background,
            payload.user_id,
            payload.prompt
        )

        return ApiResponse(
            success=True,
            message="Image generation started in background"
        )
    except Exception as e:
        logger.error(f"Image generation error for user {payload.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_image_background(user_id: str, prompt: str):
    """Background task for image generation"""
    try:
        session = manager.get_user_session(user_id)
        # Implement image generation logic here
        logger.info(f"🎨 Generating image for user {user_id}: {prompt}")
        # This would call your image generation logic
    except Exception as e:
        logger.error(f"Background image generation failed for {user_id}: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_users": len(manager.user_sessions)
    }


@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    return {
        "active_sessions": len(manager.user_sessions),
        "total_users": len(manager.user_sessions),
        "memory_usage": "Not implemented",  # Add actual memory monitoring
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )