import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from diffusers import FluxPipeline
from PIL import Image
import json
import os
from datetime import datetime
import glob
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()

LANGUAGES = {
    "1": ("English", "en"),
    "2": ("French", "fr"),
    "3": ("German", "de"),
    "4": ("Spanish", "es"),
}

class RoleplayAssistant:
    def __init__(self, json_path):
        self.json_path = json_path
        self.story_state = None
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.message_count = 0
        self.image_count = 0

    
    def choose_language(self):
        print("Please choose your language:")
        for key, (name, _) in LANGUAGES.items():
            print(f"{key}. {name}")
        
        choice = input("> ").strip()
        return LANGUAGES.get(choice, ("English", "en"))[1]  # default to English

    def login_to_huggingface(self):
        token = os.getenv("HF_TOKEN")
        if token:
            subprocess.run(
                ["huggingface-cli", "login", "--token", token, "--add-to-git-credential"],
                check=True
            )
        else:
            raise EnvironmentError("HF_TOKEN not found in environment.")

    def load_models(self):
        model_name = "Qwen/Qwen3-4B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )


        self.llm = Llama.from_pretrained(
            repo_id="TheBloke/MythoMax-L2-13B-GGUF",
            filename="mythomax-l2-13b.Q4_K_M.gguf",
        )

        #image generator Balck forest
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

        # summarization
        self.pipe_sum = pipeline("summarization", model="facebook/bart-large-cnn")

        #translate
        self.translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")



    def translation_to_eng(self, input_text):
        if self.story_state['Language'] == "en":
            return input_text
        tokenizer = self.translation_tokenizer
        tokenizer.src_lang = self.story_state['Language']
        tokenizer.tgt_lang = "en"
        encoded = tokenizer(input_text, return_tensors="pt")
        generated = self.translation_model.generate(**encoded)
        return tokenizer.decode(generated[0], skip_special_tokens=True)

    def translation_from_eng(self, input_text):
        if self.story_state['Language'] == "en":
            return input_text  # no need to translate
        tokenizer = self.translation_tokenizer
        tokenizer.src_lang = 'en'
        tokenizer.tgt_lang = self.story_state['Language']
        encoded = tokenizer(input_text, return_tensors="pt")
        generated = self.translation_model.generate(**encoded)
        return tokenizer.decode(generated[0], skip_special_tokens=True)


    def json_file(self):
        if os.path.exists(self.json_path):
            user_input = input('Do you want to create a new story ? : ')
            if user_input.lower().strip() == 'yes':
                self.story_state = {
                    'Language': None,
                    'System Character': None,
                    'User Character': None,
                    'Situation': None,
                    'chat': [],
                    'Summary of the situation': None,
                }
            else:
                with open(self.json_path, "r") as f:
                    self.story_state = json.load(f)
        else:
            self.story_state = {
                'Language': None,
                'System Character': None,
                'User Character': None,
                'Situation': None,
                'chat': [],
                'Summary of the situation': None,
            }
            with open(self.json_path, "w") as f:
                json.dump(self.story_state, f, indent=2)

    def save_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.story_state, f, indent=2)
    
    def model_output_init(self, messages):

        """
        Generate a model reply based on a list of messages in OpenAI-style chat format.
        Each message should be a dict with 'role' ('user', 'system', or 'assistant') and 'content'.
        """
         
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
    
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=300)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content


    def model_output(self, messages):

        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
        output = self.llm(prompt, max_tokens=512, stop=["user:", "system:"], echo=False)
        content = output["choices"][0]["text"].strip()
        return content


    def init_story(self):
        chat_history = []

        if self.story_state['Language'] is None:
            self.story_state['Language'] = self.choose_language()

        language = self.story_state['Language']
        chat_history.append({"role": "system", "content": f"The user has chosen {language}. Let's create a character and story together by answer following question: "})

        if self.story_state['User Character'] is None:
            chat_history.append({"role": "system", "content": "First, ask the user if he can introduction himself for the story"})
            reply = self.model_output_init(chat_history)
            print(reply)
            user_reply = input("> ")
            user_reply = self.translation_to_eng(user_reply)
            chat_history.append({"role": "assistant", "content": reply})
            chat_history.append({"role": "user", "content": user_reply})
            self.story_state['User Character'] = user_reply

        if self.story_state['System Character'] is None:
            chat_history.append({"role": "system", "content": "Now ask the user who he want to interact with"})
            reply = self.model_output_init(chat_history)
            print(reply)
            user_reply = input("> ")
            user_reply = self.translation_to_eng(user_reply)
            chat_history.append({"role": "assistant", "content": reply})
            chat_history.append({"role": "user", "content": user_reply})
            self.story_state['System Character'] = user_reply

        if self.story_state['Situation'] is None:
            chat_history.append({"role": "system", "content": "Finally, ask what are the situation and what stroy he want?"})
            reply = self.model_output_init(chat_history)
            print(reply)
            user_reply = input("> ")
            user_reply = self.translation_to_eng(user_reply)
            chat_history.append({"role": "assistant", "content": reply})
            chat_history.append({"role": "user", "content": user_reply})
            self.story_state['Situation'] = user_reply

        self.save_json()



    def summarize_chat(self):
        if not self.story_state.get('chat'):
            print('⚠️ Error: no chat history to summarize.')
            return

        # Generate summary
        chat_text = " ".join([msg["content"] for msg in self.story_state['chat'][-5:]])
        summary = self.pipe_sum(chat_text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        self.story_state['Summary of the situation'] = summary
        self.save_json()




        # Create a backup with no chat
        backup = self.story_state.copy()
        backup['chat'] = []  # Strip chat history for smaller file

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(self.json_path)[0]
        backup_filename = f"{base_name}_backup_{timestamp}.json"

        with open(backup_filename, "w") as f:
            json.dump(backup, f, indent=2)

        print(f"✅ Summary saved. Backup created: {backup_filename}")

        # Keep only 5 latest backups
        backup_files = sorted(glob.glob(f"{base_name}_backup_*.json"), reverse=True)
        for old_backup in backup_files[5:]:
            os.remove(old_backup)
            print(f"🗑️ Removed old backup: {old_backup}")

    def chat_add(self, role, content):
        if 'chat' not in self.story_state:
            self.story_state['chat'] = []
        
        self.story_state['chat'].append({"role": role, "content": content})
        self.message_count += 1

        # Auto-summary every 5 messages
        if self.message_count >= 5:
            self.summarize_chat()
            self.message_count = 0

        self.save_json()


    def run(self):
        print('Start the process')

        # Load models and story state
        self.login_to_huggingface()
        self.load_models()
        self.json_file()
        self.init_story()

        base_prompt = f"""You are a roleplay assistant. Always respond in-character with immersive dialogue and include narrative actions (didascalies) in the style of stage directions using asterisks (*). 
        Make interactions vivid, describing facial expressions, body language, and emotional tone. If the user answer includes [text], use it to influence your response.
        You are playing the character: {self.story_state['System Character']}
        The story so far is: {self.story_state['Summary of the situation']}"""

        print("🗨️  Roleplay started. Type 'exit' to quit.")   

        chat_history = self.story_state['chat']
        

        # Main loop
        print("🗨️  Roleplay started. Type 'exit' to quit.")   

        while True:
            user_input = input("🧑 You: ")
            user_input = self.translation_to_eng(user_input)
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Ending the session.")
                break
            
            self.chat_add("user", user_input)

            messages = [
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": user_input}
            ]
            reply = self.model_output(messages)
            reply = self.translation_from_eng(reply)
            print(f"AI: {reply}")
            self.chat_add("system", reply)

                # Check for image generation
            if "show me" in user_input.lower():
                print("🎨 Generating image...")
                image_prompt = [
                        {"role": "system", "content": "You are an image prompt generator."},
                        {"role": "user", "content": f"the user ask you create a image prompt in english base on this last 5 messages:\n{self.story_state['chat'][-5:]} max 70 carateres"}
                    ]  # Or a custom rephrased prompt
                
                prompt = self.model_output_init(image_prompt)
                image = self.pipe(
                        prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=3.5,
                        num_inference_steps=50,
                        max_sequence_length=512,
                        generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
                image.show()
                self.image_count += 1 
                image.save(f"story_{self.story_state['User Character']}_{self.image_count}.png")



if __name__ == "__main__":
    assistant = RoleplayAssistant("story_state.json")
    assistant.run()





 