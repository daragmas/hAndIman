from flask import Flask, jsonify, request
from flask_cors import CORS
from model_manager import model_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS
CORS(app, origins=["http://localhost:3000"])

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


@app.route('/api/model/query', methods=['POST'])
def model_query():
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "No prompt provided"}), 400
        
        if model_manager.model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        prompt = data['prompt']
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        result = model_manager.predict(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        return jsonify({
            "prompt": prompt,
            "generated_text": result[0]['generated_text'],
            "model": model_manager.model_name
        })
    
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')


# Diagnostic loop

# Save user interactions