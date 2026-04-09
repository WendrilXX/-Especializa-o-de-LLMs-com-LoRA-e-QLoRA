"""
Configurações centralizadas para o pipeline de fine-tuning
"""

from dataclasses import dataclass
from typing import Optional

# ============================================================================
# CONFIGURAÇÃO DE DADOS
# ============================================================================

@dataclass
class DataConfig:
    """Configuração para geração de dados sintéticos"""
    
    # Domínio de aplicação
    domain: str = "assistência técnica de computadores e resolução de problemas"
    
    # Quantidade de pares a gerar
    num_samples: int = 50
    
    # Proporção de treino/teste
    train_ratio: float = 0.9
    
    # Modelo OpenAI para gerar dados
    openai_model: str = "gpt-3.5-turbo"
    
    # Temperatura (criatividade)
    temperature: float = 0.7
    
    # Caminhos de dados
    output_dir: str = "data"
    train_filename: str = "train_dataset.jsonl"
    test_filename: str = "test_dataset.jsonl"
    full_filename: str = "full_dataset.jsonl"


# ============================================================================
# CONFIGURAÇÃO DE QUANTIZAÇÃO (QLoRA)
# ============================================================================

@dataclass
class QuantizationConfig:
    """Configuração de Quantização 4-bit"""
    
    # Ativar quantização
    use_quantization: bool = True
    
    # Tipo de quantização (nf4 = NormalFloat 4-bit)
    bnb_4bit_quant_type: str = "nf4"
    
    # Dtype para computação
    bnb_4bit_compute_dtype: str = "float16"
    
    # Double quantization
    bnb_4bit_use_double_quant: bool = True


# ============================================================================
# CONFIGURAÇÃO DE LoRA
# ============================================================================

@dataclass
class LoraConfig:
    """Configuração de LoRA para fine-tuning eficiente"""
    
    # Rank das matrizes LoRA (obrigatório: 64)
    rank: int = 64
    
    # Fator de escala dos novos pesos (obrigatório: 16)
    alpha: int = 16
    
    # Dropout para regularização (obrigatório: 0.1)
    dropout: float = 0.1
    
    # Seed para reprodutibilidade
    seed: int = 42
    
    # Módulos a treinar
    target_modules: list = None
    
    # Módulos a salvar integralmente
    modules_to_save: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
        if self.modules_to_save is None:
            self.modules_to_save = ["lm_head"]


# ============================================================================
# CONFIGURAÇÃO DE TREINAMENTO
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuração de treinamento com otimizador paginado"""
    
    # Modelo base
    model_name: str = "meta-llama/Llama-2-7b-hf"
    
    # HuggingFace token (opcional)
    hf_token: Optional[str] = None
    
    # Diretório de saída
    output_dir: str = "models/llama2-finetuned"
    
    # Número de épocas (obrigatório)
    num_train_epochs: int = 3
    
    # Batch size por device
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    
    # Accumulation steps (simular batch maior)
    gradient_accumulation_steps: int = 4
    
    # Learning rate
    learning_rate: float = 2e-4
    
    # Otimizador (obrigatório: paged_adamw_32bit)
    optimizer: str = "paged_adamw_32bit"
    
    # Learning rate scheduler (obrigatório: cosine)
    lr_scheduler_type: str = "cosine"
    
    # Warmup ratio (obrigatório: 0.03 = 3%)
    warmup_ratio: float = 0.03
    
    # Gradient clipping
    max_grad_norm: float = 0.3
    
    # Checkpoint e salvamento
    save_strategy: str = "steps"
    save_steps: int = 25
    eval_strategy: str = "steps"
    eval_steps: int = 25
    save_total_limit: int = 2
    
    # Logging
    logging_steps: int = 10
    logging_dir: Optional[str] = None
    
    # Reproducibilidade
    seed: int = 42
    
    # Mixed precision
    fp16: bool = True
    
    # Max sequence length
    max_seq_length: int = 512


# ============================================================================
# CONFIGURAÇÃO DE PIPELINE
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuração completa do pipeline"""
    
    # Componentes
    data_config: DataConfig = None
    quantization_config: QuantizationConfig = None
    lora_config: LoraConfig = None
    training_config: TrainingConfig = None
    
    # Flags
    generate_data: bool = True
    run_training: bool = True
    
    def __post_init__(self):
        if self.data_config is None:
            self.data_config = DataConfig()
        if self.quantization_config is None:
            self.quantization_config = QuantizationConfig()
        if self.lora_config is None:
            self.lora_config = LoraConfig()
        if self.training_config is None:
            self.training_config = TrainingConfig()
            self.training_config.logging_dir = f"{self.training_config.output_dir}/logs"


# ============================================================================
# INSTÂNCIAS PADRÃO
# ============================================================================

# Configuração padrão completa
DEFAULT_CONFIG = PipelineConfig()


def get_config() -> PipelineConfig:
    """Retorna a configuração padrão do pipeline"""
    return DEFAULT_CONFIG


if __name__ == "__main__":
    # Exemplo de uso
    config = get_config()
    print("Configuração de Treinamento:")
    print(f"  Modelo: {config.training_config.model_name}")
    print(f"  Épocas: {config.training_config.num_train_epochs}")
    print(f"  LoRA Rank: {config.lora_config.rank}")
    print(f"  LoRA Alpha: {config.lora_config.alpha}")
    print(f"  LoRA Dropout: {config.lora_config.dropout}")
    print(f"  Otimizador: {config.training_config.optimizer}")
    print(f"  LR Scheduler: {config.training_config.lr_scheduler_type}")
    print(f"  Warmup Ratio: {config.training_config.warmup_ratio}")
