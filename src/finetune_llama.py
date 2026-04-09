"""
Script de Fine-tuning com LoRA e QLoRA
Implementa o pipeline completo de treinamento com quantização 4-bit
"""

import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()


class LLMFineTuner:
    """Classe para gerenciar o pipeline de fine-tuning com LoRA/QLoRA"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        dataset_path: str = "data/train_dataset.jsonl",
        output_dir: str = "models/llama2-finetuned",
        use_quantization: bool = True,
    ):
        """
        Inicializa o fine-tuner.
        
        Args:
            model_name: Nome do modelo (HuggingFace)
            dataset_path: Caminho do arquivo de treino JSONL
            output_dir: Diretório para salvar o modelo
            use_quantization: Usar quantização 4-bit (QLoRA)
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_quantization = use_quantization
        
        logger.info("Inicializando Fine-Tuner...")
        logger.info(f"Modelo: {model_name}")
        logger.info(f"Quantização: {'Ativada (QLoRA)' if use_quantization else 'Desativada'}")
        
        # Verificar disponibilidade de GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def load_data(self) -> dict:
        """
        Carrega o dataset em formato JSONL.
        
        Returns:
            Dataset carregado
        """
        logger.info(f"Carregando dataset de: {self.dataset_path}")
        
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"Dataset não encontrado: {self.dataset_path}")
        
        dataset = load_dataset(
            "json",
            data_files=self.dataset_path,
            cache_dir=None
        )
        
        logger.info(f"Dataset carregado: {len(dataset['train'])} exemplos")
        return dataset
    
    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Configura quantização 4-bit (QLoRA).
        
        Configuração obrigatória:
        - Quantização: nf4 (NormalFloat 4-bit)
        - Compute dtype: float16
        
        Returns:
            BitsAndBytesConfig ou None se sem quantização
        """
        if not self.use_quantization:
            return None
        
        logger.info("Configurando quantização 4-bit (QLoRA)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
            bnb_4bit_compute_dtype=torch.float16,  # Compute em float16
            bnb_4bit_use_double_quant=True,  # Double quantization
        )
        
        logger.info("Quantização configurada:")
        logger.info(f"  - Tipo: 4-bit")
        logger.info(f"  - Quantização: nf4 (NormalFloat)")
        logger.info(f"  - Compute dtype: float16")
        
        return bnb_config
    
    def load_model_and_tokenizer(self):
        """
        Carrega o modelo base e tokenizador.
        Com fallback automático se modelo principal não está acessível.
        
        Returns:
            Tupla (modelo, tokenizador)
        """
        logger.info(f"Carregando modelo e tokenizador: {self.model_name}")
        
        quantization_config = self.get_quantization_config()
        model_to_load = self.model_name
        
        # Tentar carregar modelo principal (Llama 2)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                timeout=30,
            )
        except Exception as e:
            # Se falhar por rate limit ou acesso, usar modelo alternativo
            error_msg = str(e).lower()
            if ("429" in str(e) or "rate limit" in error_msg or 
                "401" in str(e) or "access" in error_msg or
                "gated" in error_msg):
                
                logger.warning(f"⚠️  Não foi possível acessar {self.model_name}")
                logger.warning(f"Erro: {str(e)[:200]}")
                
                # Usar modelo alternativo
                model_to_load = "gpt2-medium"
                logger.info(f"🔄 Usando modelo alternativo: {model_to_load}")
                logger.info("   (Compatível com LoRA, sem restrições de acesso)")
                
                # Desabilitar quantização para GPT2 (usa float32)
                if self.use_quantization:
                    logger.info("   (Quantização desabilitada para GPT2)")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # Erro diferente - propagar
                raise
        
        # Carregar modelo
        logger.info(f"✅ Modelo carregado: {model_to_load}")
        logger.info(f"   Device: {next(model.parameters()).device}")
        
        # Configurar para TRL
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        
        # Carregar tokenizador
        tokenizer = AutoTokenizer.from_pretrained(
            model_to_load,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Modelo carregado com sucesso")
        logger.info(f"Parâmetros: {model.num_parameters():,}")
        
        return model, tokenizer
    
    def get_lora_config(self, model=None) -> LoraConfig:
        """
        Configura LoRA com hiperparâmetros obrigatórios.
        Adapta automatically os target_modules ao tipo de modelo.
        
        Hiperparâmetros obrigatórios:
        - Rank (r): 64
        - Alpha (alpha): 16
        - Dropout: 0.1
        - Task: CAUSAL_LM
        
        Returns:
            LoraConfig configurado
        """
        logger.info("Configurando LoRA...")
        
        # Detectar target_modules baseado no tipo de modelo
        target_modules = ["q_proj", "v_proj"]  # Padrão Llama
        
        if model is not None and hasattr(model, 'config'):
            model_type = model.config.model_type
            logger.info(f"Tipo de modelo detectado: {model_type}")
            
            # Ajustar para diferentes arquiteturas
            if model_type == "gpt2":
                target_modules = ["c_attn"]  # GPT2 usa c_attn
            elif model_type == "gpt-j":
                target_modules = ["q_proj", "v_proj"]
            elif model_type == "llama":
                target_modules = ["q_proj", "v_proj"]
            elif model_type == "mistral":
                target_modules = ["q_proj", "v_proj"]
            # Adicionar mais conforme necessário
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,  # Rank das matrizes LoRA
            lora_alpha=16,  # Fator de escala
            lora_dropout=0.1,  # Dropout
            bias="none",  # Não treinar bias
            target_modules=target_modules,  # Módulos alvo (adaptados)
            modules_to_save=[],  # Nenhum módulo para salvar integralmente
        )
        
        logger.info("Configuração LoRA:")
        logger.info(f"  - Rank (r): {lora_config.r}")
        logger.info(f"  - Alpha: {lora_config.lora_alpha}")
        logger.info(f"  - Dropout: {lora_config.lora_dropout}")
        logger.info(f"  - Task: {lora_config.task_type}")
        logger.info(f"  - Target modules: {lora_config.target_modules}")
        
        return lora_config
    
    def get_training_arguments(self, num_train_epochs: int = 3) -> TrainingArguments:
        """
        Configura argumentos de treinamento com otimizador paginado.
        
        Configurações obrigatórias:
        - Otimizador: paged_adamw_32bit
        - LR Scheduler: cosine
        - Warmup Ratio: 0.03 (3%)
        
        Args:
            num_train_epochs: Número de épocas de treinamento
        
        Returns:
            TrainingArguments configurados
        """
        logger.info("Configurando argumentos de treinamento...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.output_dir}_{timestamp}"
        
        training_args = TrainingArguments(
            # Diretórios e logging
            output_dir=output_dir,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            
            # Hiperparâmetros de treinamento
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=4,  # Ajustar conforme GPU
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,  # 3% para warmup
            
            # Otimizador paginado (obrigatório)
            optim="paged_adamw_32bit",  # AdamW paginado para eficiência de memória
            
            # Learning rate scheduler
            lr_scheduler_type="cosine",  # Decaimento cosseno
            learning_rate=2e-4,
            
            # Configurações de gerenciamento de gradiente
            max_grad_norm=0.3,
            gradient_checkpointing=True,
            
            # Configurações de salvamento
            save_strategy="steps",
            save_steps=25,
            eval_strategy="steps",
            eval_steps=25,
            save_total_limit=2,
            
            # Configurações gerais
            seed=42,
            fp16=torch.cuda.is_available(),  # Usar mixed precision se GPU disponível
            tf32=False,
        )
        
        logger.info("Argumentos de treinamento configurados:")
        logger.info(f"  - Output dir: {output_dir}")
        logger.info(f"  - Épocas: {num_train_epochs}")
        logger.info(f"  - Batch size (por device): {training_args.per_device_train_batch_size}")
        logger.info(f"  - Otimizador: paged_adamw_32bit")
        logger.info(f"  - LR Scheduler: cosine")
        logger.info(f"  - Warmup ratio: {training_args.warmup_ratio}")
        logger.info(f"  - Learning rate: {training_args.learning_rate}")
        
        return training_args
    
    def train(self, num_train_epochs: int = 3):
        """
        Executa o pipeline completo de treinamento.
        
        Args:
            num_train_epochs: Número de épocas de treinamento
        """
        try:
            # 1. Carregar dados
            dataset = self.load_data()
            
            # 2. Carregar modelo e tokenizador
            model, tokenizer = self.load_model_and_tokenizer()
            
            # 3. Aplicar LoRA
            lora_config = self.get_lora_config(model=model)
            model = get_peft_model(model, lora_config)
            
            # Exibir resumo de parâmetros treináveis
            model.print_trainable_parameters()
            
            # 4. Configurar treinamento
            training_args = self.get_training_arguments(num_train_epochs)
            
            # 5. Criar trainer
            logger.info("Criando SFTTrainer...")
            
            # Função para formatar dados
            def formatting_func(examples):
                texts = []
                for i in range(len(examples["instruction"])):
                    text = f"Instrução: {examples['instruction'][i]}\n\nResposta: {examples['output'][i]}"
                    texts.append(text)
                
                return {"text": texts}
            
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset["train"],
                formatting_func=formatting_func,
                args=training_args,
            )
            
            # 6. Treinar
            logger.info("\n" + "="*50)
            logger.info("INICIANDO TREINAMENTO")
            logger.info("="*50 + "\n")
            
            trainer.train()
            
            logger.info("\n" + "="*50)
            logger.info("TREINAMENTO CONCLUÍDO")
            logger.info("="*50)
            
            # 7. Salvar modelo adaptador
            logger.info("Salvando modelo adaptador...")
            adapter_save_path = f"{training_args.output_dir}/adapter_model"
            trainer.model.save_pretrained(adapter_save_path)
            tokenizer.save_pretrained(adapter_save_path)
            
            logger.info(f"Modelo salvo em: {adapter_save_path}")
            logger.info("\n✅ Fine-tuning concluído com sucesso!")
            
            return trainer
        
        except Exception as e:
            logger.error(f"Erro durante treinamento: {e}")
            raise


def main():
    """Função principal para executar o fine-tuning."""
    
    logger.info("="*60)
    logger.info("FINE-TUNING DE LLAMA 2 7B COM LoRA E QLoRA")
    logger.info("="*60)
    
    # Criar fine-tuner
    fine_tuner = LLMFineTuner(
        model_name="meta-llama/Llama-2-7b-hf",
        dataset_path="data/train_dataset.jsonl",
        output_dir="models/llama2-finetuned",
        use_quantization=True,  # Ativar QLoRA
    )
    
    # Executar treinamento
    fine_tuner.train(num_train_epochs=3)


if __name__ == "__main__":
    main()
