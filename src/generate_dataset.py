"""
Script de Geração de Dataset Sintético com OpenAI API GPT
Gera pares de instrução-resposta para fine-tuning de LLMs
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
import logging

from openai import OpenAI

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

# Configurar cliente OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("❌ OPENAI_API_KEY não configurada em .env")
    sys.exit(1)

client = OpenAI(api_key=api_key)
logger.info("✅ Cliente OpenAI GPT configurado")


def generate_instruction_response_pairs(
    domain: str,
    num_samples: int = 50,
    model: str = "gpt-3.5-turbo"
) -> List[Dict[str, str]]:
    """
    Gera pares de instrução-resposta usando OpenAI API.
    
    Args:
        domain: Domínio de aplicação (ex: "assistência técnica", "educação", etc.)
        num_samples: Número de pares a gerar (default: 50)
        model: Modelo OpenAI a usar (default: gpt-3.5-turbo)
    
    Returns:
        Lista de dicionários com 'instruction' e 'output'
    """
    
    dataset = []
    
    # Prompt para gerar diversas instruções
    system_prompt = f"""Você é um especialista em criar datasets de treinamento para modelos de linguagem.
Gere instruções e respostas no domínio: {domain}

Cada instrução deve ser uma pergunta ou tarefa clara e específica.
Cada resposta deve ser completa, informativa e de alta qualidade.

Retorne EXATAMENTE um JSON válido por linha, com a estrutura:
{{"instruction": "...", "output": "..."}}

Certifique-se de que cada linha é um JSON válido e separado por quebra de linha."""

    logger.info(f"Iniciando geração de {num_samples} pares no domínio: {domain}")
    
    # Gerar em lotes para evitar limite de tokens
    batch_size = 5
    batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(batches):
        batch_count = min(batch_size, num_samples - batch_idx * batch_size)
        
        user_prompt = f"Gere {batch_count} pares de instrução-resposta diferentes, inovadores e de qualidade alta. Cada linha deve ser um JSON válido."
        
        try:
            logger.info(f"Processando lote {batch_idx + 1}/{batches}...")
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                top_p=0.95
            )
            response_text = response.choices[0].message.content
            
            # Extrair JSONs da resposta
            for line in response_text.strip().split('\n'):
                if line.strip():
                    try:
                        pair = json.loads(line)
                        if "instruction" in pair and "output" in pair:
                            dataset.append(pair)
                            logger.debug(f"Adicionado par {len(dataset)}: {pair['instruction'][:50]}...")
                    except json.JSONDecodeError:
                        logger.warning(f"Falha ao decodificar JSON: {line[:100]}")
                        continue
        
        except Exception as e:
            logger.error(f"❌ Erro ao chamar OpenAI API GPT: {e}")
            sys.exit(1)
    
    logger.info(f"✅ Total de pares gerados: {len(dataset)}")
    return dataset[:num_samples]


def split_dataset(
    dataset: List[Dict[str, str]],
    train_ratio: float = 0.9
) -> tuple:
    """
    Divide o dataset em treino e teste.
    
    Args:
        dataset: Lista de pares instrução-resposta
        train_ratio: Proporção de treino (default: 0.9 = 90%)
    
    Returns:
        Tupla (train_set, test_set)
    """
    split_idx = int(len(dataset) * train_ratio)
    return dataset[:split_idx], dataset[split_idx:]


def save_jsonl(
    data: List[Dict[str, str]],
    filepath: Path
) -> None:
    """
    Salva dados em formato JSONL.
    
    Args:
        data: Lista de dicionários
        filepath: Caminho do arquivo JSONL
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Arquivo salvo: {filepath} ({len(data)} registros)")


def main():
    """Função principal para gerar o dataset."""
    
    # Configuração
    DOMAIN = "assistência técnica de computadores e resolução de problemas"
    NUM_SAMPLES = 50
    OUTPUT_DIR = Path(__file__).parent.parent / "data"
    
    # Gerar dataset
    logger.info("=== Iniciando Geração de Dataset ===")
    dataset = generate_instruction_response_pairs(
        domain=DOMAIN,
        num_samples=NUM_SAMPLES,
        model="gpt-3.5-turbo"
    )
    
    # Dividir em treino e teste
    train_set, test_set = split_dataset(dataset, train_ratio=0.9)
    
    logger.info(f"\nDivisão do dataset:")
    logger.info(f"  Treino: {len(train_set)} pares (90%)")
    logger.info(f"  Teste:  {len(test_set)} pares (10%)")
    
    # Salvar em formato JSONL
    save_jsonl(train_set, OUTPUT_DIR / "train_dataset.jsonl")
    save_jsonl(test_set, OUTPUT_DIR / "test_dataset.jsonl")
    save_jsonl(dataset, OUTPUT_DIR / "full_dataset.jsonl")
    
    logger.info("\n=== Dataset gerado com sucesso! ===")


if __name__ == "__main__":
    main()
