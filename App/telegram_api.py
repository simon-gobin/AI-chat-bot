import os
import json
import aiohttp
import asyncio
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, CallbackContext, filters
)
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
REQUEST_TIMEOUT = 30  # seconds

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

# API endpoints
ENDPOINTS = {
    "chat": f"{API_BASE_URL}/chat",
    "status": f"{API_BASE_URL}/status",
    "reset": f"{API_BASE_URL}/reset",
    "language": f"{API_BASE_URL}/language",
    "health": f"{API_BASE_URL}/health"
}

# Language mappings
LANGUAGES = {
    "en": {"flag": "🇬🇧", "name": "English"},
    "fr": {"flag": "🇫🇷", "name": "Français"},
    "es": {"flag": "🇪🇸", "name": "Español"},
    "de": {"flag": "🇩🇪", "name": "Deutsch"}
}


class APIClient:
    """Async HTTP client for API communication"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def make_request(self, method: str, url: str, json_data: dict = None) -> dict:
        """Make HTTP request with error handling"""
        try:
            session = await self.get_session()

            if method.upper() == "GET":
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                async with session.post(url, json=json_data) as response:
                    response.raise_for_status()
                    return await response.json()

        except aiohttp.ClientTimeout:
            logger.error(f"Request timeout to {url}")
            raise Exception("⏱️ Request timed out. Please try again.")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise Exception("🔌 Connection error. Please try again later.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception("❌ An unexpected error occurred.")


# Global API client
api_client = APIClient()


def get_user_id(update: Update) -> str:
    """Extract user ID from update"""
    return str(update.effective_user.id)


async def send_typing_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send typing indicator"""
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )


# 🌍 Start command with language selection
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    try:
        # Check if API is available
        await api_client.make_request("GET", ENDPOINTS["health"])

        keyboard = []
        for lang_code, lang_info in LANGUAGES.items():
            button = InlineKeyboardButton(
                f"{lang_info['flag']} {lang_info['name']}",
                callback_data=f"lang_{lang_code}"
            )
            keyboard.append([button])

        reply_markup = InlineKeyboardMarkup(keyboard)
        welcome_text = (
            "🤖 Welcome to the AI Roleplay Assistant!\n\n"
            "🌍 Please choose your preferred language to begin:"
        )
        await update.message.reply_text(welcome_text, reply_markup=reply_markup)

    except Exception as e:
        logger.error(f"Start command error: {e}")
        await update.message.reply_text(
            "❌ Bot is currently unavailable. Please try again later."
        )


# 📥 Handle language selection
async def select_language(update: Update, context: CallbackContext):
    """Handle language selection callback"""
    query = update.callback_query
    await query.answer()

    try:
        lang_code = query.data.replace("lang_", "")
        user_id = get_user_id(update)

        if lang_code not in LANGUAGES:
            await query.edit_message_text("❌ Invalid language selection.")
            return

        # Update language via API
        response = await api_client.make_request(
            "POST",
            ENDPOINTS["language"],
            {"user_id": user_id, "lang_code": lang_code}
        )

        if response.get("success"):
            lang_name = LANGUAGES[lang_code]["name"]
            success_text = (
                f"✅ Language set to {lang_name}!\n\n"
                "🎭 You can now begin your roleplay adventure. "
                "Just start typing to create your story!\n\n"
                "💡 Commands:\n"
                "/reset - Start a new story\n"
                "/status - Check current story state\n"
                "/help - Show help message"
            )
            await query.edit_message_text(success_text)
        else:
            await query.edit_message_text("❌ Failed to set language. Please try again.")

    except Exception as e:
        logger.error(f"Language selection error: {e}")
        await query.edit_message_text("❌ Error setting language. Please try again.")


# 🔄 Reset conversation
async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /reset command"""
    try:
        user_id = get_user_id(update)
        await send_typing_action(update, context)

        response = await api_client.make_request(
            "POST",
            ENDPOINTS["reset"],
            {"message": "reset", "user_id": user_id}
        )

        if response.get("success"):
            await update.message.reply_text(
                "🔄 Story has been reset! You can start a new adventure.\n"
                "Type anything to begin creating your new story."
            )
        else:
            await update.message.reply_text("❌ Failed to reset story. Please try again.")

    except Exception as e:
        logger.error(f"Reset error: {e}")
        await update.message.reply_text(str(e))


# 📊 Show bot status
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    try:
        user_id = get_user_id(update)
        await send_typing_action(update, context)

        response = await api_client.make_request(
            "GET",
            f"{ENDPOINTS['status']}/{user_id}"
        )

        if response.get("success"):
            state = response.get("data", {})

            # Format status message
            status_parts = ["📊 **Current Story Status:**\n"]

            if state.get("Language"):
                lang_name = LANGUAGES.get(state["Language"], {}).get("name", state["Language"])
                status_parts.append(f"🌍 Language: {lang_name}")

            if state.get("User Character"):
                status_parts.append(f"👤 Your Character: {state['User Character']}")

            if state.get("System Character"):
                status_parts.append(f"🤖 AI Character: {state['System Character']}")

            if state.get("Situation"):
                status_parts.append(f"📖 Story: {state['Situation'][:100]}...")

            if state.get("chat"):
                msg_count = len(state["chat"])
                status_parts.append(f"💬 Messages: {msg_count}")

            if not any([state.get("User Character"), state.get("System Character")]):
                status_parts.append("🎭 Story not started yet. Send a message to begin!")

            status_text = "\n".join(status_parts)
            await update.message.reply_text(status_text, parse_mode="Markdown")
        else:
            await update.message.reply_text("❌ Could not retrieve status.")

    except Exception as e:
        logger.error(f"Status error: {e}")
        await update.message.reply_text(str(e))


# ❓ Help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
🤖 **AI Roleplay Assistant Help**

🎭 **How to use:**
• Start chatting to create your roleplay story
• The bot will guide you through character creation
• Use natural language to interact with AI characters
• Say "show me" to generate images of scenes

📋 **Commands:**
/start - Choose language and restart
/reset - Start a new story
/status - Check your current story state
/help - Show this help message

🌟 **Tips:**
• Be descriptive in your responses
• Use [actions] to influence the story
• The AI remembers your conversation context
• Each user has their own separate story

❓ **Need help?** Just start typing to begin your adventure!
    """
    await update.message.reply_text(help_text, parse_mode="Markdown")


# 💬 Handle messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages"""
    try:
        user_message = update.message.text
        user_id = get_user_id(update)

        # Show typing indicator
        await send_typing_action(update, context)

        # Send message to API
        response = await api_client.make_request(
            "POST",
            ENDPOINTS["chat"],
            {"message": user_message, "user_id": user_id}
        )

        if response.get("success"):
            reply = response.get("data", {}).get("reply", "❌ No response received")

            # Split long messages
            if len(reply) > 4096:  # Telegram message limit
                for i in range(0, len(reply), 4096):
                    await update.message.reply_text(reply[i:i + 4096])
            else:
                await update.message.reply_text(reply)
        else:
            error_msg = response.get("error", "Unknown error occurred")
            await update.message.reply_text(f"❌ Error: {error_msg}")

    except Exception as e:
        logger.error(f"Message handling error: {e}")
        await update.message.reply_text(str(e))


# 🚨 Error handler
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler"""
    logger.error(f"Exception while handling an update: {context.error}")

    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "❌ An unexpected error occurred. Please try again or use /start to restart."
        )


# 🤖 Bot main
async def main():
    """Main bot function"""
    try:
        # Create application
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

        # Add handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CallbackQueryHandler(select_language, pattern="^lang_"))
        app.add_handler(CommandHandler("reset", reset))
        app.add_handler(CommandHandler("status", status))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        # Add error handler
        app.add_error_handler(error_handler)

        logger.info("🤖 Bot started successfully!")

        # Run bot
        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        # Keep running
        await asyncio.Event().wait()

    except Exception as e:
        logger.error(f"Bot startup error: {e}")
        raise
    finally:
        # Cleanup
        await api_client.close()
        if 'app' in locals():
            await app.stop()
            await app.shutdown()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise