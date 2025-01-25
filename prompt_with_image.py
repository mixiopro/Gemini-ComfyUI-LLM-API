import numpy as np
from PIL import Image
import io
import base64
from typing import Literal
from litellm import completion

PROMPT_SUFFIX = " Provide only the answer without additional explanation or commentary."

# Vision models that support image input
VISION_MODELS = {
    "gpt-4-vision-preview": "gpt-4-vision-preview",
    "gpt-4o": "gpt-4o", 
    "claude-3-sonnet": "anthropic/claude-3-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "gemini-pro-vision": "vertex_ai/gemini-pro-vision",
    "llava": "ollama/llava" 
}

# Text chat models
TEXT_MODELS = {
    # OpenAI models
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    
    # Claude models
    "claude-2.1": "anthropic/claude-2.1",
    "claude-2.0": "anthropic/claude-2.0",
    "claude-instant": "anthropic/claude-instant-1.2",
    
    # Together AI models
    "llama-2-7b": "together_ai/togethercomputer/Llama-2-7B-32K-Instruct",
    
    # Clarifai models
    "gpt-4-clarifai": "clarifai/openai.chat-completion.GPT-4",
    "gpt-3.5-turbo-clarifai": "clarifai/openai.chat-completion.GPT-3_5-turbo"
}

class PromptException(Exception):
    """Custom exception for prompt-related errors"""
    pass

class PromptWithImage:
    """ComfyUI node for LLM chat completion with optional image input"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Combine all model options
        all_models = list(VISION_MODELS.keys()) + list(TEXT_MODELS.keys())
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_type": (all_models,),
                "api_key": ("STRING", {"multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 100, "min": 1, "max": 4096}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "LLM"

    def process(
        self,
        prompt: str,
        model_type: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        image: np.ndarray = None,
        seed: int = 0
    ):
        """Process the prompt with optional image using LiteLLM"""
        
        if not api_key:
            raise ValueError("API key is required")

        # Check if image is provided but model doesn't support vision
        if image is not None and model_type not in VISION_MODELS:
            raise PromptException(
                f"Model {model_type} does not support image input. Please use one of: {', '.join(VISION_MODELS.keys())}"
            )

        # Get the full model name based on the selected type
        model = VISION_MODELS.get(model_type) or TEXT_MODELS.get(model_type)
        
        # Prepare messages
        messages = []
        if image is not None:
            # Convert from numpy RGBA to PIL Image
            pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            # Convert to RGB if necessary
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            # Convert to base64
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + PROMPT_SUFFIX},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt + PROMPT_SUFFIX}]
        
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                api_key=api_key,
                drop_params=True
            )
            return (response.choices[0].message.content,)

        except Exception as e:
            raise PromptException(f"LiteLLM error: {str(e)}")