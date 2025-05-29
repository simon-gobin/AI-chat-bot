import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import FluxPipeline
from llama_cpp import Llama
import json
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    BUSY = "busy"


@dataclass
class ModelInfo:
    name: str
    status: ModelStatus
    model: Optional[Any] = None
    error: Optional[str] = None


class AsyncModelManager:
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent model operations
        self.model_queue = asyncio.Queue(maxsize=1)  # Prevent multiple heavy operations

    async def load_model_async(self, model_name: str, model_type: str) -> bool:
        """Load a model asynchronously without blocking the main thread"""
        try:
            self.models[model_name] = ModelInfo(model_name, ModelStatus.LOADING)

            # Use executor to run model loading in separate thread
            if model_type == "qwen":
                model = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._load_qwen_model
                )
            elif model_type == "llama":
                model = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._load_llama_model
                )
            elif model_type == "flux":
                model = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._load_flux_model
                )
            elif model_type == "summarization":
                model = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._load_summarization_model
                )
            elif model_type == "translation":
                model = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._load_translation_model
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.models[model_name] = ModelInfo(model_name, ModelStatus.READY, model)
            logger.info(f"✅ {model_name} loaded successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to load {model_name}: {str(e)}"
            self.models[model_name] = ModelInfo(model_name, ModelStatus.ERROR, error=error_msg)
            logger.error(error_msg)
            return False

    def _load_qwen_model(self):
        """Load Qwen model in separate thread"""
        try:
            model_name = "Qwen/Qwen3-4B"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading Qwen: {e}")
            raise

    def _load_llama_model(self):
        """Load Llama model in separate thread"""
        try:
            return Llama.from_pretrained(
                repo_id="TheBloke/MythoMax-L2-13B-GGUF",
                filename="mythomax-l2-13b.Q4_K_M.gguf",
                n_ctx=2048,  # Limit context to save memory
                n_threads=4  # Limit threads
            )
        except Exception as e:
            logger.error(f"Error loading Llama: {e}")
            raise

    def _load_flux_model(self):
        """Load FLUX model in separate thread"""
        try:
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16,
                device_map="balanced"
            )
            pipe.enable_model_cpu_offload()
            pipe.safety_checker = None
            pipe.requires_safety_checker = False
            return pipe
        except Exception as e:
            logger.error(f"Error loading FLUX: {e}")
            raise

    def _load_summarization_model(self):
        """Load summarization model in separate thread"""
        try:
            return pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            logger.error(f"Error loading summarization model: {e}")
            raise

    def _load_translation_model(self):
        """Load translation model in separate thread"""
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
            model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
            tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading translation model: {e}")
            raise

    async def generate_text_async(self, model_name: str, messages: list, max_tokens: int = 300) -> Optional[str]:
        """Generate text asynchronously"""
        model_info = self.models.get(model_name)
        if not model_info or model_info.status != ModelStatus.READY:
            logger.warning(f"Model {model_name} not ready. Status: {model_info.status if model_info else 'Not loaded'}")
            return None

        try:
            # Mark model as busy
            model_info.status = ModelStatus.BUSY

            # Add to queue to prevent multiple concurrent operations
            await self.model_queue.put(model_name)

            if model_name == "qwen":
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._generate_qwen, model_info.model, messages, max_tokens
                )
            elif model_name == "llama":
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._generate_llama, model_info.model, messages, max_tokens
                )
            else:
                result = None

            # Mark model as ready again
            model_info.status = ModelStatus.READY

            # Remove from queue
            await self.model_queue.get()

            return result

        except Exception as e:
            logger.error(f"Error generating text with {model_name}: {e}")
            model_info.status = ModelStatus.READY
            await self.model_queue.get()
            return None

    def _generate_qwen(self, model_data, messages, max_tokens):
        """Generate text with Qwen model"""
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        return tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    def _generate_llama(self, model, messages, max_tokens):
        """Generate text with Llama model"""
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
        output = model(prompt, max_tokens=max_tokens, stop=["user:", "system:"], echo=False)
        return output["choices"][0]["text"].strip()

    async def generate_image_async(self, prompt: str) -> Optional[str]:
        """Generate image asynchronously"""
        model_info = self.models.get("flux")
        if not model_info or model_info.status != ModelStatus.READY:
            return None

        try:
            model_info.status = ModelStatus.BUSY
            await self.model_queue.put("flux")

            image = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._generate_image, model_info.model, prompt
            )

            model_info.status = ModelStatus.READY
            await self.model_queue.get()

            return image

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            model_info.status = ModelStatus.READY
            await self.model_queue.get()
            return None

    def _generate_image(self, pipe, prompt):
        """Generate image with FLUX model"""
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        # Save image
        image_path = f"generated_image_{len(os.listdir('.'))}.png"
        image.save(image_path)
        return image_path

    def get_model_status(self) -> Dict[str, str]:
        """Get status of all models"""
        return {name: info.status.value for name, info in self.models.items()}

    async def initialize_all_models(self):
        """Initialize all models with proper error handling"""
        models_to_load = [
            ("qwen", "qwen"),
            ("llama", "llama"),
            ("flux", "flux"),
            ("summarization", "summarization"),
            ("translation", "translation")
        ]

        # Load models one by one to avoid memory issues
        for model_name, model_type in models_to_load:
            logger.info(f"Loading {model_name}...")
            success = await self.load_model_async(model_name, model_type)
            if not success:
                logger.warning(f"Failed to load {model_name}, continuing with other models...")

            # Small delay between models to prevent memory spikes
            await asyncio.sleep(2)


# Usage example for FastAPI integration
class AsyncRoleplayAssistant:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.model_manager = AsyncModelManager()
        self.story_state = None

    async def initialize(self):
        """Initialize the assistant and all models"""
        await self.model_manager.initialize_all_models()
        self.load_story_state()

    def load_story_state(self):
        """Load story state from JSON"""
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
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

    async def respond_async(self, user_input: str) -> Dict[str, Any]:
        """Process user input asynchronously"""
        # Handle story setup or roleplay response
        if self.story_state.get('Situation') is None:
            return await self._handle_setup(user_input)
        else:
            return await self._handle_roleplay(user_input)

    async def _handle_setup(self, user_input: str) -> Dict[str, Any]:
        """Handle story setup phase"""
        # Setup logic here
        return {"response": "Setup response", "status": "setup"}

    async def _handle_roleplay(self, user_input: str) -> Dict[str, Any]:
        """Handle roleplay phase"""
        messages = [
            {"role": "system", "content": f"You are {self.story_state['System Character']}"},
            {"role": "user", "content": user_input}
        ]

        # Try different models if one fails
        response = await self.model_manager.generate_text_async("qwen", messages)
        if not response:
            response = await self.model_manager.generate_text_async("llama", messages)

        if not response:
            response = "Sorry, I'm having trouble generating a response right now."

        return {"response": response, "status": "roleplay"}

# Example usage:
# assistant = AsyncRoleplayAssistant("story_state.json")
# await assistant.initialize()
# response = await assistant.respond_async("Hello!")