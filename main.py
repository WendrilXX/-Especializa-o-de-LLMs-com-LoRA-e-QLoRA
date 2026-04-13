#!/usr/bin/env python
"""
Script Principal - Orquestra o Pipeline Completo
Executa em sequência: setup → dados → fine-tuning → inferência
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command: list, description: str) -> bool:
    """
    Executa um comando no terminal.
    
    Args:
        command: Lista com comando e argumentos
        description: Descrição do que está sendo executado
    
    Returns:
        True se sucesso, False se erro
    """
    logger.info("\n" + "="*60)
    logger.info(f"{description}")
    logger.info("="*60)
    
    try:
        result = subprocess.run(command, check=True)
        logger.info(f"{description} - concluído com sucesso\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao executar {description}: {e}\n")
        return False
    except FileNotFoundError as e:
        logger.error(f"Comando não encontrado: {e}\n")
        return False


def main():
    """Função principal"""
    
    parser = argparse.ArgumentParser(
        description="Pipeline Completo de Fine-tuning com LoRA/QLoRA"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Executar verificação de setup"
    )
    
    parser.add_argument(
        "--data",
        action="store_true",
        help="Gerar dataset sintético"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Executar fine-tuning"
    )
    
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Modo de inferência (interativo)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executar pipeline completo (setup → dados → train)"
    ) 
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Pular verificações de setup"
    )
    
    args = parser.parse_args()
    
    # Se nenhuma opção especificada, mostrar help
    if not any([args.setup, args.data, args.train, args.infer, args.all]):
        parser.print_help()
        return 1
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE DE FINE-TUNING COM LoRA/QLoRA")
    logger.info("="*60 + "\n")
    
    # 1. Setup
    if args.setup or args.all:
        if not args.skip_checks:
            success = run_command(
                [sys.executable, "setup.py"],
                "Verificação de Ambiente"
            )
            if not success and not args.skip_checks:
                logger.error("Setup falhou. Corrija os erros acima.")
                return 1
    
    # 2. Gerar Dados
    if args.data or args.all:
        success = run_command(
            [sys.executable, "src/generate_dataset.py"],
            "Geração de Dataset Sintético"
        )
        if not success:
            logger.error("Geração de dados falhou.")
            return 1
    
    # 3. Fine-tuning
    if args.train or args.all:
        success = run_command(
            [sys.executable, "src/finetune_simple.py"],
            "Fine-tuning com LoRA/QLoRA"
        )
        if not success:
            logger.error("Fine-tuning falhou.")
            return 1
        logger.info("\nPipeline de treinamento concluído com sucesso!")
        logger.info("Para testar o modelo, execute: python main.py --infer\n")
    
    # 4. Inferência
    if args.infer:
        success = run_command(
            [sys.executable, "src/inference.py"],
            "Modo de Inferência"
        )
        if not success:
            logger.error("Inferência falhou.")
            return 1
    
    # Se apenas --all, mostrar resumo
    if args.all:
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
        logger.info("="*60)
        logger.info("\nArquivos gerados:")
        logger.info("  | data/train_dataset.jsonl")
        logger.info("  | data/test_dataset.jsonl")
        logger.info("  | models/tinyllama-finetuned_*")
        logger.info("\nPara testar o modelo:")
        logger.info("  python main.py --infer\n")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
