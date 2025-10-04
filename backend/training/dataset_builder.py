import json
import os
from typing import List, Dict
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


class DiagnosticTreeDatasetBuilder:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.tree_data: Dict = {}
        self.load_tree()
    
    def load_tree(self):
        """Load the diagnostic tree from JSON"""
        try:
            # Handle relative path from backend/training/ to data/
            json_path = self.json_path
            if not os.path.isabs(json_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                json_path = os.path.join(script_dir, "..", "..", json_path)
                json_path = os.path.normpath(json_path)
            
            with open(json_path, 'r') as f:
                self.tree_data = json.load(f)
            
            # Validate that we loaded data
            if not self.tree_data:
                raise ValueError("Loaded empty diagnostic tree")
            
            problem = self.tree_data.get('problem', 'Unknown problem')
            logger.info(f"Loaded diagnostic tree for problem: {problem}")
            
        except FileNotFoundError as e:
            logger.error(f"Diagnostic tree file not found: {json_path}")
            self.tree_data = {}
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in diagnostic tree: {e}")
            self.tree_data = {}
            raise
        except Exception as e:
            logger.error(f"Failed to load diagnostic tree: {e}")
            self.tree_data = {}
            raise
    
    def generate_training_examples(self) -> List[Dict[str, str]]:
        """
        Generate training examples from the diagnostic tree.
        
        Creates examples for:
        1. Initial problem assessment
        2. Each diagnostic question and interpretation
        3. Hypothesis support/falsification
        4. Final diagnosis and recommendations
        
        Returns list of dicts with 'text' field in instruction format
        """
        examples = []
        
        problem = self.tree_data.get('problem', 'unknown issue')
        hypotheses = self.tree_data.get('hypotheses', [])
        safety_checks = self.tree_data.get('safety_checks', [])
        
        # 1. Generate initial problem statement examples
        examples.extend(self._generate_initial_examples(problem, safety_checks))
        
        # 2. Generate diagnostic question examples
        examples.extend(self._generate_question_examples(problem, hypotheses))
        
        # 3. Generate hypothesis evaluation examples
        examples.extend(self._generate_hypothesis_examples(problem, hypotheses))
        
        # 4. Generate diagnosis and fix recommendation examples
        examples.extend(self._generate_diagnosis_examples(problem, hypotheses))
        
        # 5. Generate conversational rephrasing examples
        examples.extend(self._generate_rephrasing_examples(hypotheses))
        
        logger.info(f"Generated {len(examples)} training examples")
        return examples
    
    def _format_prompt(self, instruction: str, response: str) -> str:
        """Format training data in instruction-following format for Mistral"""
        return f"""<s>[INST] You are a helpful handyman diagnostic assistant. You help users diagnose and fix common household problems by asking strategic questions and providing clear, actionable solutions.

{instruction} [/INST] {response}</s>"""
    
    def _generate_initial_examples(self, problem: str, safety_checks: List[str]) -> List[Dict[str, str]]:
        """Generate examples for initial problem assessment and safety warnings"""
        examples = []
        
        # Example 1: User states problem
        instruction = f"A user says: 'My {problem}. Can you help me figure out what's wrong?'"
        response = "I can help you diagnose this issue. Before we start, let me share some important safety information:\n"
        for i, check in enumerate(safety_checks, 1):
            response += f"{i}. {check}\n"
        response += "\nNow, let me ask you some diagnostic questions to narrow down the cause."
        
        examples.append({"text": self._format_prompt(instruction, response)})
        
        # Example 2: Direct problem statement
        instruction = f"Help me fix: {problem}"
        response = f"I'll help you diagnose why your {problem}. First, please ensure you follow these safety precautions, then I'll guide you through some tests."
        
        examples.append({"text": self._format_prompt(instruction, response)})
        
        return examples
    
    def _generate_question_examples(self, problem: str, hypotheses: List[Dict]) -> List[Dict[str, str]]:
        """Generate examples for asking diagnostic questions"""
        examples = []
        
        for hypothesis in hypotheses:
            for test in hypothesis.get('tests', []):
                question = test['question']
                
                # Example: Rephrasing technical question conversationally
                instruction = f"The user is experiencing: {problem}. Ask them this diagnostic question in a friendly way: {question}"
                response = f"Let me check something: {question}"
                
                examples.append({"text": self._format_prompt(instruction, response)})
                
                # Example: Explaining why we're asking
                instruction = f"Why are you asking: {question}"
                response = f"This test helps us determine if the issue is related to {hypothesis['name']}. {hypothesis['description']}"
                
                examples.append({"text": self._format_prompt(instruction, response)})
        
        return examples
    
    def _generate_hypothesis_examples(self, problem: str, hypotheses: List[Dict]) -> List[Dict[str, str]]:
        """Generate examples for hypothesis support/falsification based on test results"""
        examples = []
        
        for hypothesis in hypotheses:
            hyp_name = hypothesis['name']
            description = hypothesis['description']
            
            for test in hypothesis.get('tests', []):
                question = test['question']
                expected = test['expected_outcome']
                supports = test.get('supports', [])
                falsifies = test.get('falsifies', [])
                
                # Example: Answer matches expected (supports hypothesis)
                instruction = f"For the problem '{problem}', I asked: '{question}' and the user answered '{expected}'. What does this tell us?"
                response = f"Based on that answer, this supports the hypothesis that the issue is: {hyp_name}. {description}"
                if falsifies:
                    response += f" This also helps us rule out: {', '.join(falsifies)}."
                
                examples.append({"text": self._format_prompt(instruction, response)})
                
                # Example: Answer doesn't match (falsifies hypothesis)
                opposite = "no" if expected.lower() == "yes" else "yes"
                instruction = f"For the problem '{problem}', I asked: '{question}' and the user answered '{opposite}'. What does this mean?"
                response = f"That answer suggests it's probably not {hyp_name}. Let's investigate other possibilities."
                
                examples.append({"text": self._format_prompt(instruction, response)})
        
        return examples
    
    def _generate_diagnosis_examples(self, problem: str, hypotheses: List[Dict]) -> List[Dict[str, str]]:
        """Generate examples for final diagnosis and fix recommendations"""
        examples = []
        
        for hypothesis in hypotheses:
            hyp_name = hypothesis['name']
            description = hypothesis['description']
            fix = hypothesis.get('fix', 'Consult a professional.')
            
            # Example: Providing diagnosis
            instruction = f"Based on our diagnostic questions, the most likely cause of '{problem}' is: {hyp_name}. Provide the diagnosis and solution."
            response = f"Diagnosis: {hyp_name}\n\n{description}\n\nRecommended fix: {fix}"
            
            examples.append({"text": self._format_prompt(instruction, response)})
            
            # Example: User asks "what should I do?"
            instruction = f"I've determined my {problem} is caused by {hyp_name}. What should I do to fix it?"
            response = f"Here's how to fix {hyp_name}:\n\n{fix}\n\nTake your time and follow the steps carefully. Let me know if you need clarification on any part."
            
            examples.append({"text": self._format_prompt(instruction, response)})
            
            # Example: Asking for more details about the fix
            instruction = f"Can you explain more about how to fix {hyp_name}?"
            response = f"{fix}\n\nRemember to work carefully and stop if anything seems unsafe."
            
            examples.append({"text": self._format_prompt(instruction, response)})
        
        return examples
    
    def _generate_rephrasing_examples(self, hypotheses: List[Dict]) -> List[Dict[str, str]]:
        """Generate examples for rephrasing technical questions conversationally"""
        examples = []
        
        conversational_pairs = [
            ("Run a thin blade along sash edges — does it lift free easily?", 
             "Try running a thin blade (like a putty knife) along the edges where the window meets the frame. Does the window come free easily?"),
            ("After freeing edges, does window move but scrape heavily?", 
             "Now that you've freed the edges, when you try to open the window, does it move but scrape or drag against the frame?"),
            ("Does the sash drop when released, as if no support?", 
             "If you lift the window up and then let go, does it drop down quickly like there's nothing holding it up?"),
            ("Look along tracks — do you see dirt or screws blocking the path?", 
             "Take a look at the tracks where the window slides. Do you see any dirt, debris, or screws sticking out that might be blocking it?"),
        ]
        
        for technical, conversational in conversational_pairs:
            instruction = f"Rephrase this technical question in a friendly, conversational way: {technical}"
            response = conversational
            
            examples.append({"text": self._format_prompt(instruction, response)})
        
        return examples
    
    def create_dataset(self) -> Dataset:
        """Create HuggingFace Dataset from examples"""
        examples = self.generate_training_examples()
        dataset = Dataset.from_list(examples)
        logger.info(f"Created dataset with {len(dataset)} examples")
        return dataset
    
    def get_tree_context(self) -> str:
        """Get tree context for model prompts"""
        if not self.tree_data:
            return ""
        
        problem = self.tree_data.get('problem', 'Unknown')
        hypotheses = self.tree_data.get('hypotheses', [])
        
        context = f"Problem: {problem}\n\n"
        context += "Possible causes:\n"
        
        for i, hyp in enumerate(hypotheses, 1):
            context += f"{i}. {hyp['name']}: {hyp['description']}\n"
        
        return context
    
    def save_dataset(self, output_path: str = "data/training_dataset"):
        """Save the dataset to disk"""
        dataset = self.create_dataset()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save as JSON for easy inspection
        dataset.to_json(os.path.join(output_path, "dataset.json"))
        
        # Also save in HuggingFace format
        dataset.save_to_disk(output_path)
        
        logger.info(f"Dataset saved to {output_path}")
        return dataset


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Build dataset from window diagnostics
    builder = DiagnosticTreeDatasetBuilder("data/window_diagnostic.json")
    
    # Generate and inspect examples
    examples = builder.generate_training_examples()
    
    print(f"\nGenerated {len(examples)} training examples\n")
    print("Sample examples:\n")
    print("="*80)
    
    # Show first 3 examples
    for i, example in enumerate(examples[:3], 1):
        print(f"\nExample {i}:")
        print(example['text'][:500] + "..." if len(example['text']) > 500 else example['text'])
        print("-"*80)
    
    # Create and save dataset
    dataset = builder.create_dataset()
    print(f"\nDataset created with {len(dataset)} examples")
    
    # Optionally save
    # builder.save_dataset("data/training_dataset")
