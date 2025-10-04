from flask import Flask, jsonify, request
from flask_cors import CORS
from model_manager import model_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS
CORS(app, origins=["http://localhost:3000"])
# Request models
class TextInput(BaseModel):
    text: str

class ModelLoadRequest(BaseModel):
    model_name: str
    task: str = "sentiment-analysis"

# Load default model on startup
def init_model():
    logger.info("Starting up... Loading default model")
    model_manager.load_model()

# Call init after app is created
with app.app_context():
    init_model()

@app.route('/')
def home():
    return jsonify({"message":'Welcome to hAndIman'})


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Check current model status"""
    return jsonify({
        "loaded": model_manager.model is not None,
        "model_name": model_manager.model_name
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
# handleUserInput route

# Diagnostic loop

# Save user interactions