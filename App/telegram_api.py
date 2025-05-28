import os
import json
import requests
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, CallbackContext, filters
)

API_URL = "http://localhost:8000/chat"
STATUS_URL = "http://localhost:8000/status"
RESET_URL = "http://localhost:8000/reset"
STORY_STATE_FILE = "/app/story_state.json"  # Update if needed for Docker volume

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Make sure this is set

# 🌍 Start command with language selection
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🇬🇧 English", callback_data="lang_en")],
        [InlineKeyboardButton("🇫🇷 Français", callback_data="lang_fr")],
        [InlineKeyboardButton("🇪🇸 Español", callback_data="lang_es")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("🌍 Please choose a language:", reply_markup=reply_markup)

# 📥 Handle language choice
async def select_language(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()

    lang_code = query.data.replace("lang_", "")
    lang_label = {"en": "English", "fr": "Français", "es": "Español"}[lang_code]

    # Call FastAPI to update the language
    user_id = update.effective_user.id
    requests.post("http://localhost:8000/language", json={"telegram_id": user_id, "lang_code": lang_code})
    if response.status_code == 200:
        await query.edit_message_text(text=f"✅ Language set to {lang_label}! You can now begin chatting.")
    else:
        await query.edit_message_text(text="❌ Failed to set language.")

# 🔄 Reset conversation
async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    requests.post(RESET_URL)
    await update.message.reply_text("🔄 Story state has been reset.")

# 📊 Show bot status
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = requests.get(STATUS_URL)
    state = res.json()
    status_text = "📊 Current state:\n" + "\n".join([f"{k}: {v}" for k, v in state.items() if k != "chat"])
    await update.message.reply_text(status_text)

# 💬 Handle messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    res = requests.post(API_URL, json={"message": user_message})
    reply = res.json().get("reply", "❌ No response")
    await update.message.reply_text(reply)

# 🧠 Bot main
if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(select_language, pattern="^lang_"))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot running...")
    app.run_polling()
