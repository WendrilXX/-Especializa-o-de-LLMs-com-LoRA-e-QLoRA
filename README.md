# Fine-tuning de GPT-2 Medium com LoRA e QLoRA

Pipeline completo de fine-tuning de modelos de linguagem fundacionais utilizando técnicas de eficiência de parâmetros (PEFT/LoRA) e quantização (QLoRA) para viabilizar o treinamento em hardwares limitados (validado em CPU).

## Política de Integridade Acadêmica

**NOTA IMPORTANTE**: Partes geradas/complementadas com IA (GitHub Copilot, Claude Haiku 3.5), revisadas por estudante.

Conforme contrato pedagógico: Qualquer uso de IA foi revisado criticamente. O código foi testado, validado e todos os componentes foram verificados para garantir funcionalidade e conformidade com os requisitos do laboratório.

## Objetivo

Construir um pipeline profissional que demonstra:
- Geração de datasets sintéticos com OpenAI API
- Quantização 4-bit (QLoRA) com `bitsandbytes`
- LoRA com parâmetros configuráveis (`peft`)
- Treinamento eficiente com Trainer nativo do HuggingFace
- Otimizador adamw_torch com scheduler cosine para convergência suave

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

**Nota sobre GPU:**
- CUDA 11.8+ recomendado para melhor desempenho
- Verificar: `nvidia-smi`

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

**Configuração implementada em `finetune_llama.py`:**

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.float16, # Compute em float16
    bnb_4bit_use_double_quant=True,
)
```

**Benefícios:**
- Reduz uso de memória GPU em ~75%
- Mantém qualidade de treinamento
- Viabiliza treino em GPUs com 8GB

## Passo 3: Arquitetura LoRA

**Configuração implementada em `finetune_llama.py`:**

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                              # Rank das matrizes
    lora_alpha=16,                     # Fator de escala
    lora_dropout=0.1,                  # Dropout para regularização
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["lm_head"],
)
```

**Parâmetros Treináveis:**
- ~3.3M parâmetros LoRA (vs 7B do modelo completo)
- Redução de ~99.95% em relação a full fine-tuning

## Passo 4: Pipeline de Treinamento

### Comando

```bash
python src/finetune_llama.py
```

### Configurações de Treinamento

```python
TrainingArguments(
    # Otimizador: paged_adamw_32bit
    optim="paged_adamw_32bit",  # Reduz picos de memória GPU→CPU
    
    # Learning Rate Scheduler: cosine
    lr_scheduler_type="cosine",  # Decaimento suave
    warmup_ratio=0.03,           # Aquecimento 3% inicial
    
    # Hiperparâmetros
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
)
```

### Saída Esperada
207/207 [45:30<00:00, 13.25s/it]
Epoca 1/3 - Loss: 7.137
Epoca 2/3 - Loss: 5.824
Epoca 3/3 - Loss: 4.513 (37% de redução)

Modelo adaptador salvo em:
→ models/gpt2-lora_20260408_214954
Modelo adaptador salvo em:
→ models/llama2-finetuned_20240308_143056/adapter_model
```

### Monitoramento

```bash
# Ver logs em tempo real
tensorboard --logdir models/llama2-finetuned_*/logs
```Resultados do Treinamento

Após 3 épocas de fine-tuning:

```
models/gpt2-lora_20260408_214954/
├── adapter_model/
│   ├── adapter_config.json      # Config LoRA
│   ├── pytorch_model.bin        # Adaptador treinado (~6.2MB)
│   └── tokenizer.model          # Tokenizador
├── checkpoint-68/               # Checkpoints intermediários
└── checkpoint-69/

Estatísticas:
  - Modelo: GPT-2 Medium
  - Parâmetros base: 354.8M
  - Parâmetros treináveis (LoRA): 6.2M (1.74%)
  - Épocas: 3
  - Loss inicial: 7.137
  - Loss final: 4.513 (37% de redução)
  - Tempo total: ~45 minutos (CPU
└── checkpoint-25/
    └── ... (checkpoints intermediários)
```
 LoRA
model_path = "models/gpt2-lora_20260408_214954/adapter_model"
model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # CPU ou GPU automaticamente
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
print(tokenizer.decode(outputs[0], skip_special_tokens=True
tokenizer = AutoTokenizer.from_pretrained(
    "models/llama2-finetuned_20240308_143056/adapter_model"
)

# Fazer predição
input_text = "Como atualizar os drivers?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Requisitos do Laboratório

### Engenharia de Dados Sintéticos
- [x] Script Python com OpenAI API
- [x] 50+ pares instrução-resposta
- [x] Divisão 90% treino / 10% teste
- [x] Formato JSONL

### Quantização
- [x] BitsAndBytesConfig configurado
- [x] 4-bit com nf4
- [x] compute_dtype = float16

### LoRA
- [x] LoraConfig com CAUSAL_LM
- [x] Rank (r) = 64
- [x] Alpha = 16
- [x] Dropout = 0.1

### Treinamento
- [x] Trainer nativo com LoRA implementado
- [x] Otimizador: adamw_torch
- [x] LR Scheduler: cosine
- [x] Warmup Steps: 1
- [x] 3 Épocas completadas (loss: 7.14 → 4.51)
- [x] Modelo fine-tuned salvo com adapter + tokenizer

## Política de Integridade Acadêmica

### Atribuição de IA

**Partes geradas/complementadas com IA, revisadas por [Wendril Gabriel]**

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
# Editar .env com sua chave
```

### Erro: "CUDA out of memory"
```python
# No finetune_llama.py, reduzir batch size
per_device_train_batch_size=2  # ao invés de 4
```

### Modelo muito lento para baixar
```bash
# Usar snapshot_download para retomar de onde parou
huggingface-cli download meta-llama/Llama-2-7b-hf
```

## Referências

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Quantization with bitsandbytes](https://huggingface.co/docs/bitsandbytes)
- [LoRA Paper](https://arxiv.org/abs/2106.09714)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)


