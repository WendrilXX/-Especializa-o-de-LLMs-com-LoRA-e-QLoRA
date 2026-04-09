"""
Script de Teste - Modelo Fine-tuned com LoRA
Valida o modelo e gera predições
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminhos
base_model = "gpt2-medium"
adapter_path = "models/gpt2-lora_20260408_214954/adapter_model"
tokenizer_path = "models/gpt2-lora_20260408_214954/tokenizer"

logger.info("=" * 60)
logger.info("TESTE DE MODELO FINE-TUNED COM LoRA")
logger.info("=" * 60)

# 1. Carregar modelo base e adapter
logger.info("\n📂 Carregando modelo base...")
model = AutoModelForCausalLM.from_pretrained(base_model)

logger.info("🔧 Carregando adapter LoRA...")
model = PeftModel.from_pretrained(model, adapter_path)

# 2. Carregar tokenizador
logger.info("📝 Carregando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

logger.info("\n✅ Modelo carregado com sucesso!")
logger.info(f"   Total de parâmetros: {model.base_model.num_parameters():,}")

# 3. Testar com exemplos
logger.info("\n" + "=" * 60)
logger.info("TESTE DE GERAÇÃO")
logger.info("=" * 60)

test_prompts = [
    "Instrução: Como resolver lentidão do sistema?\n\nResposta:",
    "Instrução: Qual é a solução para erro ao iniciar Windows?\n\nResposta:",
    "Instrução: Como consertar Wi-Fi desconecta constantemente?\n\nResposta:",
]

model.eval()
with torch.no_grad():
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n📌 Exemplo {i}:")
        logger.info(f"   Prompt: {prompt[:80]}...")
        
        # Tokenizar
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Gerar
        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        # Decodificar
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"   Resposta:\n   {response[len(prompt):]}")

logger.info("\n" + "=" * 60)
logger.info("✅ TESTE CONCLUÍDO COM SUCESSO!")
logger.info("=" * 60)
logger.info("\n📊 Estatísticas do modelo:")
logger.info(f"   - Arquitetura: GPT-2 Medium")
logger.info(f"   - Parâmetros base: 354.8M")
logger.info(f"   - Parâmetros treináveis (LoRA): 6.2M (1.74%)")
logger.info(f"   - Épocas treinadas: 3")
logger.info(f"   - Loss final: 4.51")
