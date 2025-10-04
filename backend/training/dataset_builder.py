import json
from typing import List, Dict
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

class DiagnosticTreeDatasetBuilder:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.tree_data = None
        self.load_tree()
    
    def load_tree(self):
        """Load the diagnostic tree from JSON"""
        try:
            with open(self.json_path, 'r') as f:
                self.tree_data = json.load(f)
            logger.info(f"Loaded diagnostic tree: {self.tree_data.get('tree_name')}")
        except Exception as e:
            logger.error(f"Failed to load diagnostic tree: {e}")
            raise
    
    def generate_training_examples(self) -> List[Dict[str, str]]:
        """
        Generate training examples from the diagnostic tree
        
        Returns list of dicts with 'text' field in instruction format
        """
        examples = []
        
        # Use explicit training examples if provided
        if 'training_examples' in self.tree_data:
            for example in self.tree_data['training_examples']:
                prompt = self._format_prompt(example['input'], example['output'])
                examples.append({"text": prompt})
        
        # Generate examples from tree structure
        examples.extend(self._generate_from_tree())
        
        logger.info(f"Generated {len(examples)} training examples")
        return examples
    
    def _format_prompt(self, instruction: str, response: str) -> str:
        """Format training data in instruction-following format"""
        return f"""<s>[INST] You are a medical diagnostic assistant. Based on the symptoms provided, follow the diagnostic tree to provide appropriate guidance.

{instruction} [/INST] {response}</s>"""
    
    def _generate_from_tree(self) -> List[Dict[str, str]]:
        """Generate training examples by traversing the tree"""
        examples = []
        nodes = self.tree_data.get('nodes', [])
        
        for node in nodes:
            if 'responses' in node:
                for response_type, response_data in node['responses'].items():
                    if response_data.get('diagnosis'):
                        # Create training example
                        instruction = f"Patient scenario related to: {node['question']} Answer: {response_type}"
                        response = f"Diagnosis: {response_data['diagnosis']}"
                        
                        prompt = self._format_prompt(instruction, response)
                        examples.append({"text": prompt})
        
        return examples
    
    def create_dataset(self) -> Dataset:
        """Create HuggingFace Dataset from examples"""
        examples = self.generate_training_examples()
        dataset = Dataset.from_list(examples)
        return dataset
    
    def get_tree_context(self) -> str:
        """Get tree context for model prompts"""
        if not self.tree_data:
            return ""
        
        context = f"Diagnostic Tree: {self.tree_data.get('tree_name', 'Unknown')}\n"
        context += f"Version: {self.tree_data.get('version', 'Unknown')}\n\n"
        
        # Add node summaries
        for node in self.tree_data.get('nodes', []):
            context += f"- {node.get('question', '')}\n"
        
        return context