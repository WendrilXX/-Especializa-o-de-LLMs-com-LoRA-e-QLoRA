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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not configured in .env")
    sys.exit(1)

client = OpenAI(api_key=api_key)
logger.info("OpenAI GPT client configured")


def generate_instruction_response_pairs(
    domain: str,
    num_samples: int = 50,
    model: str = "gpt-3.5-turbo"
) -> List[Dict[str, str]]:
    """
    Gera pares de instrução-resposta usando a API OpenAI.
    
    Args:
        domain: Domínio da aplicação (ex: "suporte técnico", "educação")
        num_samples: Número de pares a gerar (padrão: 50)
        model: Modelo OpenAI a usar (padrão: gpt-3.5-turbo)
    
    Returns:
        Lista de dicionários com 'instruction' e 'output'
    """
    
    dataset = []
    
    # Prompt do sistema para gerar pares instrução-resposta
    system_prompt = f"""Você é um especialista em criar datasets de treinamento para modelos de linguagem.
Gere instruções e respostas no domínio: {domain}

Cada instrução deve ser uma pergunta ou tarefa clara e específica.
Cada resposta deve ser completa, informativa e de alta qualidade.

Retorne EXATAMENTE um JSON válido por linha, com a estrutura:
{{"instruction": "...", "output": "..."}}

Certifique-se de que cada linha é um JSON válido e separado por quebra de linha."""

    logger.info(f"Iniciando geração de {num_samples} pares no domínio: {domain}")
    
    # Gerar em lotes para evitar limites de tokens
    batch_size = 5
    batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(batches):
        batch_count = min(batch_size, num_samples - batch_idx * batch_size)
        
        user_prompt = f"Gere {batch_count} pares instrução-resposta diferentes, inovadores e de alta qualidade. Cada linha deve ser um JSON válido."
        
        try:
            logger.info(f"Processing batch {batch_idx + 1}/{batches}...")
            
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
                            logger.debug(f"Par adicionado {len(dataset)}: {pair['instruction'][:50]}...")
                    except json.JSONDecodeError:
                        logger.warning(f"Falha ao decodificar JSON: {line[:100]}")
                        continue
        
        except Exception as e:
            logger.error(f"Erro ao chamar API OpenAI: {e}")
            sys.exit(1)
    
    logger.info(f"Total de pares gerados: {len(dataset)}")
    return dataset[:num_samples]


def split_dataset(
    dataset: List[Dict[str, str]],
    train_ratio: float = 0.9
) -> tuple:
    """
    Split dataset into train and test.
    
    Args:
        dataset: List of instruction-response pairs
        train_ratio: Proportion for training (default: 0.9 = 90%)
    
    Returns:
        Tuple (train_set, test_set)
    """
    split_idx = int(len(dataset) * train_ratio)
    return dataset[:split_idx], dataset[split_idx:]


def save_jsonl(
    data: List[Dict[str, str]],
    filepath: Path
) -> None:
    """
    Save data in JSONL format.
    
    Args:
        data: List of dictionaries
        filepath: Path to JSONL file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"File saved: {filepath} ({len(data)} records)")


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
