from .prompt_with_image import PromptWithImage  # Import your actual node class

NODE_CLASS_MAPPINGS = {
    "PromptWithImage": PromptWithImage  # The key is what will show up in ComfyUI interface
}

# Optional: If your node has display names different from class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptWithImage": "LLM Chat (Image Optional)"  # Optional, for better UI display
}

# Required: To properly register your node
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
