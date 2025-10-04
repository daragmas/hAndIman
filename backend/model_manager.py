from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_name = None
        
    def load_model(self, model_name="mistral7b", task="sentiment-analysis"):
        try:
            logger.info(f"Loading model: {model_name}")
            self.model = pipeline(task, model=model_name)
            self.model_name = model_name
            logger.info(f"Model loaded successfully: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, text):
        """Run inference on the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            result = self.model(text)
            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Global model instance
model_manager = ModelManager()