import json
import os
from typing import Optional, Dict, List, Tuple
from model_manager import model_manager


class DiagnosticEngine:
    def __init__(self, json_path: str):
        # Handle relative path from backend/training/ to data/
        if not os.path.isabs(json_path):
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels (from backend/training/ to root) then into data/
            json_path = os.path.join(script_dir, "..", "..", json_path)
            json_path = os.path.normpath(json_path)
        
        with open(json_path, "r") as f:
            self.tree = json.load(f)
        
        # Track hypothesis confidence scores
        self.hypothesis_scores = {
            h["name"]: 0.0 for h in self.tree["hypotheses"]
        }
        
        # Track which hypotheses are still viable
        self.active_hypotheses = set(h["name"] for h in self.tree["hypotheses"])
        
        # Track conversation history for context
        self.conversation_history = []
        
        # Track which tests have been performed
        self.tests_performed = []
        
        self.reset()
    
    def reset(self):
        """Reset the diagnostic session"""
        self.hypothesis_scores = {
            h["name"]: 0.0 for h in self.tree["hypotheses"]
        }
        self.active_hypotheses = set(h["name"] for h in self.tree["hypotheses"])
        self.conversation_history = []
        self.tests_performed = []
    
    def display_safety_checks(self):
        """Display safety warnings before starting diagnosis"""
        checks = self.tree.get("safety_checks", [])
        if checks:
            print("\n" + "="*60)
            print("⚠️  SAFETY CHECKS - PLEASE READ BEFORE STARTING ⚠️")
            print("="*60)
            for i, check in enumerate(checks, 1):
                print(f"{i}. {check}")
            print("="*60 + "\n")
    
    def get_next_best_test(self) -> Optional[Tuple[Dict, Dict]]:
        """
        Find the next most discriminating test to perform.
        Returns (hypothesis, test) tuple or None if no tests remain.
        """
        best_test: Optional[Dict] = None
        best_score = -1
        best_hypothesis: Optional[Dict] = None
        
        for hypothesis in self.tree["hypotheses"]:
            # Skip if hypothesis already falsified
            if hypothesis["name"] not in self.active_hypotheses:
                continue
            
            for test in hypothesis.get("tests", []):
                # Skip if test already performed
                test_id = (hypothesis["name"], test["question"])
                if test_id in self.tests_performed:
                    continue
                
                # Score based on severity and how many hypotheses it affects
                severity = test.get("severity", 0.5)
                supports_count = len(test.get("supports", []))
                falsifies_count = len(test.get("falsifies", []))
                
                # Prefer tests that can falsify multiple hypotheses
                # or have high severity
                discriminating_power = (falsifies_count * 2 + supports_count) * severity
                
                if discriminating_power > best_score:
                    best_score = discriminating_power
                    best_test = test
                    best_hypothesis = hypothesis
        
        if best_test is not None and best_hypothesis is not None:
            return (best_hypothesis, best_test)
        return None
    
    def interpret_user_answer(self, user_input: str, test: Dict) -> str:
        """
        Use the LLM to interpret fuzzy user responses and normalize to yes/no.
        """
        expected = test["expected_outcome"].strip().lower()
        user_lower = user_input.strip().lower()
        
        # Direct matches
        if user_lower in ["yes", "y", "yeah", "yep", "yup", "true"]:
            return "yes"
        if user_lower in ["no", "n", "nope", "nah", "false"]:
            return "no"
        
        # Use LLM for ambiguous responses
        prompt = f"""The user was asked: "{test['question']}"
They responded: "{user_input}"

Is their answer essentially "yes" or "no"? Respond with only the word "yes" or "no"."""
        
        try:
            response = model_manager.predict(prompt, max_length=10)[0]["generated_text"]
            # Extract yes/no from response
            response_lower = response.strip().lower()
            if "yes" in response_lower:
                return "yes"
            elif "no" in response_lower:
                return "no"
        except Exception as e:
            print(f"[Warning: Could not interpret answer via LLM: {e}]")
        
        # Fallback to direct comparison
        return "yes" if user_lower == expected else "no"
    
    def update_hypotheses(self, test: Dict, answer: str):
        """
        Update hypothesis scores and active set based on test result.
        """
        expected = test["expected_outcome"].strip().lower()
        severity = test.get("severity", 0.5)
        
        if answer == expected:
            # Answer matches expectation - support these hypotheses
            for hyp_name in test.get("supports", []):
                if hyp_name in self.hypothesis_scores:
                    self.hypothesis_scores[hyp_name] += severity
            
            # Falsify the listed hypotheses
            for hyp_name in test.get("falsifies", []):
                if hyp_name in self.active_hypotheses:
                    self.active_hypotheses.remove(hyp_name)
                    print(f"  [Ruled out: {hyp_name}]")
        else:
            # Answer doesn't match - falsify supported hypotheses
            for hyp_name in test.get("supports", []):
                if hyp_name in self.active_hypotheses:
                    self.active_hypotheses.remove(hyp_name)
                    print(f"  [Ruled out: {hyp_name}]")
    
    def get_diagnosis(self) -> Optional[Dict]:
        """
        Return the most likely hypothesis based on current evidence.
        """
        if not self.active_hypotheses:
            return None
        
        # Find the active hypothesis with highest score
        best_hypothesis = None
        best_score = -1
        
        for hypothesis in self.tree["hypotheses"]:
            if hypothesis["name"] in self.active_hypotheses:
                score = self.hypothesis_scores[hypothesis["name"]]
                if score > best_score:
                    best_score = score
                    best_hypothesis = hypothesis
        
        return best_hypothesis
    
    def format_question_conversationally(self, question: str) -> str:
        """
        Use LLM to rephrase question in a friendly, conversational way.
        """
        prompt = f"""You are a helpful handyman assistant. Rephrase this diagnostic question in a friendly, conversational way:

Original: {question}

Keep it clear and simple. Just provide the rephrased question, nothing else."""
        
        try:
            response = model_manager.predict(prompt, max_length=100)[0]["generated_text"]
            # Clean up the response
            conversational = response.strip()
            # Remove any quotation marks the model might add
            conversational = conversational.strip('"\'')
            return conversational
        except Exception as e:
            print(f"[Warning: Could not rephrase question: {e}]")
            return question
    
    def run(self):
        """
        Main diagnostic loop - interactive CLI version.
        """
        self.reset()
        print(f"\n🔧 Handyman Diagnostic Assistant")
        print(f"Problem: {self.tree['problem']}")
        
        self.display_safety_checks()
        
        print("Let's figure out what's wrong. I'll ask you a few questions.\n")
        print("(You can answer with 'yes', 'no', or describe what you observe)\n")
        
        question_count = 0
        max_questions = 10  # Safety limit
        
        while question_count < max_questions:
            # Get next best test
            next_test_info = self.get_next_best_test()
            
            if not next_test_info:
                print("\n✓ I've asked all relevant questions.")
                break
            
            hypothesis, test = next_test_info
            question_count += 1
            
            # Mark test as performed
            test_id = (hypothesis["name"], test["question"])
            self.tests_performed.append(test_id)
            
            # Ask the question
            conversational_q = self.format_question_conversationally(test["question"])
            print(f"\n🤔 Question {question_count}: {conversational_q}")
            
            # Get user input
            user_input = input("   Your answer: ").strip()
            
            if user_input.lower() in ["quit", "exit", "stop"]:
                print("Diagnostic session ended.")
                return
            
            # Interpret answer
            interpreted = self.interpret_user_answer(user_input, test)
            
            # Update hypotheses based on answer
            self.update_hypotheses(test, interpreted)
            
            # Store in conversation history
            self.conversation_history.append({
                "question": test["question"],
                "answer": user_input,
                "interpreted": interpreted
            })
            
            # Check if we have a clear diagnosis
            if len(self.active_hypotheses) == 1:
                print("\n✓ I think I've identified the problem!")
                break
            elif len(self.active_hypotheses) == 0:
                print("\n⚠️  Hmm, I've ruled out all standard causes.")
                break
        
        # Provide diagnosis
        print("\n" + "="*60)
        print("📋 DIAGNOSIS")
        print("="*60)
        
        diagnosis = self.get_diagnosis()
        
        if diagnosis:
            print(f"\n✓ Most likely cause: {diagnosis['name']}")
            print(f"   Description: {diagnosis['description']}")
            print(f"\n🔧 Recommended fix:")
            print(f"   {diagnosis['fix']}")
            print(f"\n   Confidence score: {self.hypothesis_scores[diagnosis['name']]:.2f}")
        else:
            print("\n⚠️  Unable to determine the cause with confidence.")
            print("   This might be a more complex issue requiring professional inspection.")
        
        # Show other possibilities if any remain
        if diagnosis and len(self.active_hypotheses) > 1:
            print(f"\n   Other possibilities to consider:")
            for hyp in self.tree["hypotheses"]:
                if hyp["name"] in self.active_hypotheses and hyp["name"] != diagnosis["name"]:
                    score = self.hypothesis_scores[hyp["name"]]
                    print(f"   - {hyp['name']} (confidence: {score:.2f})")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    engine = DiagnosticEngine("data/window_diagnostic.json")
    engine.run()
