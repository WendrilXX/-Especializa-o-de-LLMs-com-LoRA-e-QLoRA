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
    
    logger.info("=" * 60)
    logger.info("FINE-TUNING COM LoRA - VERSÃO SIMPLIFICADA")
    logger.info("=" * 60)
    
    # Configurações
    model_name = "gpt2-medium"  # Usar GPT2 (sem autenticação)
    dataset_path = "data/train_dataset.jsonl"
    output_dir = f"models/gpt2-lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 1. Carregar dataset
    logger.info(f"\n📂 Carregando dataset...") 
    dataset = load_dataset("json", data_files=dataset_path, cache_dir=None)
    logger.info(f"✅ Dataset carregado: {len(dataset['train'])} exemplos")
    
    # 2. Carregar modelo e tokenizador
    logger.info(f"\n🤖 Carregando modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    logger.info(f"✅ Modelo carregado: {model.num_parameters():,} parâmetros")
    
    # 3. Preparar dados para treinamento
    def tokenize_function(examples):
        """Tokenizar dados"""
        texts = []
        for i in range(len(examples["instruction"])):
            text = f"Instrução: {examples['instruction'][i]}\n\nResposta: {examples['output'][i]}"
            texts.append(text)
        
        # Tokenizar
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors=None,
        )
        
        # Adicionar labels (igual aos inputs para language modeling)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    logger.info("\n📝 Tokenizando dataset...")
    train_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizando",
    )
    logger.info(f"✅ Dataset tokenizado: {len(train_dataset)} exemplos")
    
    # 4. Configurar LoRA
    logger.info("\n⚙️  Configurando LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["c_attn"],  # GPT2
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("✅ LoRA configurado")
    
    # 5. Configurar argumentos de treinamento
    logger.info("\n🎯 Configurando argumentos de treinamento...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=4,
        save_total_limit=2,
        logging_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        weight_decay=0.01,
        optim="adamw_torch",  # Usar adam em vez de paged_adamw
        max_grad_norm=1.0,
        seed=42,
    )
    logger.info("✅ Argumentos configurados")
    
    # 6. Criar trainer
    logger.info("\n🚀 Criando Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    logger.info("✅ Trainer criado")
    
    # 7. Treinar
    logger.info("\n" + "=" * 60)
    logger.info("🏃 INICIANDO TREINAMENTO")
    logger.info("=" * 60 + "\n")
    
    try:
        train_result = trainer.train()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 60)
        logger.info(f"Perda final: {train_result.training_loss:.4f}")
        logger.info(f"Modelo salvo em: {output_dir}")
        
        # 8. Salvar modelo
        logger.info("\n💾 Salvando modelo...")
        model.save_pretrained(f"{output_dir}/adapter_model")
        tokenizer.save_pretrained(f"{output_dir}/tokenizer")
        logger.info(f"✅ Modelo salvo com sucesso!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro durante treinamento: {str(e)}")
        logger.exception(e)
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
