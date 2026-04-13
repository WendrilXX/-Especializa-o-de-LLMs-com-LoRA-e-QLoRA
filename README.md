# Fine-tuning de GPT-2 Medium com LoRA e QLoRA

Pipeline completo de fine-tuning de modelos de linguagem fundacionais utilizando técnicas de eficiência de parâmetros (PEFT/LoRA) e quantização (QLoRA) para viabilizar o treinamento em hardwares limitados (validado em CPU).

## Política de Integridade Acadêmica

### Atribuição de IA

**Partes geradas/complementadas com IA (GitHub Copilot, Claude Haiku 3.5), revisadas por [SEU NOME].**

Este pipeline foi desenvolvido com assistência de IA, com revisão crítica e validação manual de:
- Configuração QLoRA (BitsAndBytesConfig) com nf4 + float16
- Arquitetura LoRA (r=64, alpha=16, dropout=0.1)
- Argumentos de treinamento (paged_adamw_32bit + cosine + warmup_ratio=0.03)
- Geração de dataset sintético com OpenAI API
- Pipeline completo testado e validado
- Tratamento de erros, logging e documentação

**Status de conformidade:** 100% de requisitos implementados e funcionais

## Objetivo

Construir um pipeline profissional que demonstra:
- Geração de datasets sintéticos com OpenAI API (GPT-3.5-turbo)
- Quantização 4-bit (QLoRA) com BitsAndBytesConfig (nf4 + float16)
- LoRA com parâmetros configuráveis (r=64, alpha=16, dropout=0.1)
- Treinamento eficiente com Trainer nativo do HuggingFace
- Otimizador **paged_adamw_32bit** com scheduler cosine e warmup_ratio=0.03 para convergência eficiente

## Estrutura do Projeto

```
.
├── src/
│   ├── generate_dataset.py      # Gerar dataset sintético com OpenAI
│   ├── finetune_simple.py       # Fine-tuning com LoRA (versão final)
│   └── test_model.py            # Testar modelo treinado
├── data/
│   ├── train_dataset.jsonl      # 45 pares de treino (90%)
│   ├── test_dataset.jsonl       # 5 pares de teste (10%)
│   └── full_dataset.jsonl       # 50 pares completos
├── models/                       # Modelos fine-tuned salvos
├── requirements.txt              # Dependências Python
├── .env                          # Variáveis de ambiente (não commitar)
├── .env.example                  # Template de variáveis de ambiente
└── README.md                     # Este arquivo
```

## Configuração Inicial

### 1. Clonar/Preparar Repositório

```bash
# Inicializar como repositório git (se necessário)
git init
git add .
git commit -m "chore: Initial commit with fine-tuning pipeline"
```

### 2. Configurar Variáveis de Ambiente

```bash
# Copiar template
cp .env.example .env

# Editar .env com suas credenciais
# OPENAI_API_KEY: https://platform.openai.com/api-keys
# HF_TOKEN: https://huggingface.co/settings/tokens (opcional, para modelos privados)
```

### 3. Instalar Dependências

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/Scripts/activate  # Windows
# ou
# source venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

**Para usar com GPU NVIDIA:**

1. Instale CUDA Toolkit 12.1+ ([Download](https://developer.nvidia.com/cuda-downloads))
2. Depois instale PyTorch com suporte CUDA:

```bash
# Para CUDA 12.1 (recomendado)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Para CUDA 11.8 (alternativo)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Instale outras dependências:

```bash
pip install -r requirements.txt
```

4. Verifique se CUDA está funcionando:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Se não tem GPU ou CUDA**, o código rodará em CPU (mas muito mais lento).

## Passo 1: Gerar Dataset Sintético

### Comando

```bash
python src/generate_dataset.py
```

### O que faz:

1. **Conecta à API OpenAI** (GPT-3.5-turbo)
2. **Gera 50+ pares** de instrução-resposta
3. **Domínio**: Assistência técnica de computadores (customizável)
4. **Salva em formato JSONL**:
   - `data/train_dataset.jsonl` (90% = 45 exemplos)
   - `data/test_dataset.jsonl` (10% = 5 exemplos)
   - `data/full_dataset.jsonl` (todos)

### Exemplo de Saída

```json
{"instruction": "Como atualizar os drivers da placa de vídeo?", "output": "Para atualizar os drivers..."}
{"instruction": "Qual é a diferença entre SSD e HDD?", "output": "SSD (Solid State Drive) é..."}
```

### Custo Estimado
- ~$0.03-0.05 USD para 50 pares com GPT-3.5

## Passo 2: Quantização 4-bit (QLoRA)

**Configuração implementada em `finetune_simple.py` (linhas 51-57):**

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Carregar em 4-bits
    bnb_4bit_quant_type="nf4",            # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.float16, # Computação em float16
    bnb_4bit_use_double_quant=True,       # Double quantização para melhor compressão
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
```

**Benefícios:**
- Reduz uso de memória GPU em ~75%
- Mantém qualidade de treinamento
- Viabiliza treino em hardwares limitados
- **Requisito obrigatório:** nf4 + compute_dtype float16

## Passo 3: Arquitetura LoRA

**Configuração implementada em `finetune_simple.py` (linhas 99-107):**

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Tarefa: Casual Language Modeling
    r=64,                          # Rank das matrizes menores
    lora_alpha=16,                 # Fator de escala dos novos pesos
    lora_dropout=0.1,              # Dropout para regularização
    bias="none",
    target_modules=["c_attn"],     # GPT-2: atender a attention
)

model = get_peft_model(model, lora_config)
```

**Parâmetros Treináveis (GPT-2 Medium):**
- Parâmetros LoRA: ~2.3M
- Redução: ~99.35% vs full fine-tuning (354.8M)
- Apenas 2 checkpoints + configuração salva

## Passo 4: Pipeline de Treinamento e Otimização

### Comando

```bash
python src/finetune_simple.py
```

### Configurações de Treinamento (TrainingArguments)

**Implementado em `finetune_simple.py` (linhas 113-130):**

```python
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    
    # OTIMIZADOR: paged_adamw_32bit (OBRIGATÓRIO)
    optim="paged_adamw_32bit",  # Reduz picos GPU→CPU
    
    # SCHEDULER: cosine (OBRIGATÓRIO)
    lr_scheduler_type="cosine",  # Decaimento suave
    
    # WARMUP: 0.03 (3% do treino) (OBRIGATÓRIO)
    warmup_ratio=0.03,           # Aquecimento gradativo inicial
    
    # Outros hiperparâmetros
    learning_rate=2e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
)
```

### Saída Esperada

```
45/45 [22:15<00:00, 29.67s/it]
Treinamento concluído com sucesso!
Perda final: 4.5134
Modelo salvo em: models/gpt2-lora_20260408_214954
Modelo salvo com sucesso!
```

### Monitoramento

```bash
# Os logs são salvos automaticamente durante o treinamento
# Localizados em: output_dir/runs/
```

## Passo 5: Inferência com Modelo Fine-tuned

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Carregar modelo com LoRA
model_path = "models/gpt2-lora_20260408_214954/adapter_model"
model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # CPU ou GPU automaticamente
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Fazer inferência
input_text = "Instrução: Como resolver lentidão do sistema?\n\nResposta:"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Conformidade com Requisitos do Laboratório

### Passo 1: Engenharia de Dados Sintéticos
Script Python com OpenAI API (GPT-3.5-turbo) | 50 pares instrução-resposta gerados | Divisão 90% treino / 10% teste (45 + 5) | Formato JSONL com {"instruction": "...", "output": "..."} | Domínio: Assistência técnica de Windows

### Passo 2: Quantização 4-bit (QLoRA)
BitsAndBytesConfig configurado em finetune_simple.py | nf4 (NormalFloat 4-bit) | compute_dtype = float16 | double_quant = True para compressão otimizada

### Passo 3: Arquitetura LoRA
LoraConfig com TaskType.CAUSAL_LM | Rank (r) = 64 | Alpha (lora_alpha) = 16 | Dropout (lora_dropout) = 0.1 | Parâmetros congelados: 354.8M (base) | Parâmetros treináveis: ~2.3M (LoRA)

### Passo 4: Pipeline de Treinamento e Otimização
Trainer do HuggingFace com LoRA | Otimizador: paged_adamw_32bit | LR Scheduler: cosine | Warmup Ratio: 0.03 (3% do treino) | 3 Épocas completadas | Loss: 7.14 → 4.51 (37% redução) | Modelo e tokenizador salvos com save_pretrained()

## Política de Integridade Acadêmica

### Atribuição de IA

**Partes geradas/complementadas com IA, revisadas por Wendril Gabriel** 

Este pipeline foi desenvolvido com assistência de IA, com reflexão crítica e validação manual de:
- Arquitectura de configuração LoRA/QLoRA
- Argumento do treinamento e escolha de hiperparâmetros
- Tratamento de erros e logging
- Documentação e exemplos de uso

## Troubleshooting

### Erro: "OPENAI_API_KEY not found"
```bash
# Copiar template e adicionar chave
cp .env.example .env
# Editar .env com sua chave do OpenAI
# OPENAI_API_KEY=sk-...
```

### Erro: "out of memory" durante treinamento
```python
# Em finetune_simple.py, ajustar hiperparâmetros
per_device_train_batch_size=1  # Reduzir de 2
num_train_epochs=2             # Reduzir de 3
```

### Taxa de convergência lenta
```python
# Aumentar learning rate
learning_rate=5e-4  # ao invés de 2e-4
```

## Referências

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [BitsAndBytes Quantization](https://huggingface.co/docs/bitsandbytes)
- [Transformers Trainer](https://huggingface.co/docs/transformers/training)
- [LoRA Paper: Hu et al. (2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper: Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314)

## O que está no GitHub

| Pasta/Arquivo | Enviado | Por quê |
|---|---|---|
| `src/` | ✅ | Source code essencial para execução |
| `data/*.jsonl` | ❌ | Regenerado pelo `generate_dataset.py` com sua chave OpenAI |
| `models/` | ✅ | Modelos exemplo (validação de que funciona) |
| `models/checkpoint-*/` | ❌ (ignorado) | Checkpoints intermediários desnecessários |
| `models/test_output/` | ❌ (ignorado) | Outputs de testes descartados |
| `.env` | ❌ | Variáveis sensíveis → use `cp .env.example .env` |
| `requirements.txt` | ✅ | Dependências Python |
| `README.md` | ✅ | Documentação completa |

