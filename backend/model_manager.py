from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.task = None
        
    def load_model(self, model_name="mistralai/Mistral-7B-v0.1", task="text-generation"):
        """Load a HuggingFace model"""
        try:
            logger.info(f"Loading model: {model_name} for task: {task}")
            
            # For text generation, you can add custom parameters
            if task == "text-generation":
                # Use GPU if available
                device = 0 if torch.cuda.is_available() else -1
                
                self.model = pipeline(
                    task,
                    model=model_name,
                    device=device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    # Add these for better performance
                    max_length=512,  # Default max length
                    truncation=True
                )
            else:
                self.model = pipeline(task, model=model_name)
            
            self.model_name = model_name
            self.task = task
            logger.info(f"Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, text, max_length=100, temperature=0.7, top_p=0.9, num_return_sequences=1):
        """
        Run inference on the loaded model
        
        Parameters for text generation:
        - max_length: Maximum length of generated text
        - temperature: Randomness (0.1=focused, 1.0=creative)
        - top_p: Nucleus sampling threshold
        - num_return_sequences: Number of different completions to generate
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            if self.task == "text-generation":
                result = self.model(
                    text,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True  # Enable sampling for more diverse outputs
                )
            else:
                result = self.model(text)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

model_manager = ModelManager()