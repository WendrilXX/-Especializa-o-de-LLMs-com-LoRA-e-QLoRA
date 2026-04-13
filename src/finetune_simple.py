"""
Fine-tuning Simplificado com LoRA
Versão robusta que funciona com versões recentes de transformers/trl
"""

import os
import torch
import logging
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()


def main():
    """Pipeline de fine-tuning simplificado"""
    
    logger.info("="*60)
    logger.info("Fine-tuning with LoRA - Simplified Version")
    logger.info("="*60)
    
    # Configuration
    model_name = "gpt2-medium"  # Use GPT2 (no authentication needed)
    dataset_path = "data/train_dataset.jsonl"
    output_dir = f"models/gpt2-lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 1. Load dataset
    logger.info(f"\nLoading dataset...")
    dataset = load_dataset("json", data_files=dataset_path, cache_dir=None)
    logger.info(f"Dataset loaded: {len(dataset['train'])} examples")
    
    # 2. Configure 4-bit quantization (QLoRA)
    logger.info("\nConfiguring 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("QLoRA configured with nf4, float16 and double_quant")
    
    # 3. Load model and tokenizer with quantization
    logger.info(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    logger.info(f"Modelo carregado: {model.num_parameters():,} parâmetros")
    
    # 4. Preparar dados para treinamento
    def tokenize_function(examples):
        """Tokeniza os dados de entrada"""
        texts = []
        for i in range(len(examples["instruction"])):
            text = f"Instrução: {examples['instruction'][i]}\n\nResposta: {examples['output'][i]}"
            texts.append(text)
        
        # Tokenizar entrada
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors=None,
        )
        
        # Adicionar labels (mesmos que input_ids para language modeling)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    logger.info("\nTokenizando dataset...")
    train_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizando",
    )
    logger.info(f"Dataset tokenized: {len(train_dataset)} examples")
    
    # 5. Configure LoRA
    logger.info("\nConfiguring LoRA...")  
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["c_attn"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("LoRA configured successfully")
    
    # 6. Configure training arguments
    logger.info("\nConfiguring training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=4,
        save_total_limit=2,
        logging_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        seed=42,
    )
    logger.info("Training arguments configured")
    
    # 7. Create trainer
    logger.info("\nCreating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    logger.info("Trainer created successfully")
    
    # 8. Train
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60 + "\n")
    
    try:
        train_result = trainer.train()
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Final loss: {train_result.training_loss:.4f}")
        logger.info(f"Model saved in: {output_dir}")
        
        # 9. Save model
        logger.info("\nSaving model...")
        model.save_pretrained(f"{output_dir}/adapter_model")
        tokenizer.save_pretrained(f"{output_dir}/tokenizer")
        logger.info("Model saved successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.exception(e)
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
