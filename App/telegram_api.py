import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

API_URL = "http://localhost:8000/chat"
STATUS_URL = "http://localhost:8000/status"
RESET_URL = "http://localhost:8000/reset"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Set this in your environment

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Welcome to the Roleplay Assistant! Just type to begin.")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    requests.post(RESET_URL)
    await update.message.reply_text("🔄 Story state has been reset.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = requests.get(STATUS_URL)
    state = res.json()
    status_text = f"Current state:\n" + "\n".join([f"{k}: {v}" for k, v in state.items() if k != 'chat'])
    await update.message.reply_text(status_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    res = requests.post(API_URL, json={"message": user_message})
    reply = res.json().get("reply", "❌ No response")
    await update.message.reply_text(reply)

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot running...")
    app.run_polling()
