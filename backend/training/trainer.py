import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging
import os

logger = logging.getLogger(__name__)

class DiagnosticModelTrainer:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        output_dir: str = "./models/fine_tuned"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        
    def setup_model(self, use_4bit: bool = True):
        """
        Setup model with LoRA for efficient fine-tuning
        
        use_4bit: Use 4-bit quantization to reduce memory usage
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            # Load model with quantization if specified
            if use_4bit:
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                self.model = prepare_model_for_kbit_training(self.model)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=16,  # LoRA rank
                lora_alpha=32,  # LoRA alpha
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("Model setup complete with LoRA")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def prepare_dataset(self, dataset: Dataset):
        """Tokenize the dataset"""
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        save_steps: int = 100
    ):
        """
        Train the model on the dataset
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        logger.info("Preparing dataset...")
        tokenized_dataset = self.prepare_dataset(train_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            save_steps=save_steps,
            logging_steps=10,
            save_total_limit=3,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        logger.info(f"Training complete. Model saved to {self.output_dir}")
        
        # Save final model
        self.save_model()
    
    def save_model(self, path: str = None):
        """Save the trained model"""
        save_path = path or self.output_dir
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")