"""
Script de testes e validação
Valida que todos os componentes estão funcionando corretamente
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports() -> bool:
    """Valida que todos os imports necessários estão disponíveis"""
    logger.info("\nValidando imports...")
    
    try_imports = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "peft": "PEFT",
        "trl": "TRL",
        "bitsandbytes": "BitsAndBytes",
        "openai": "OpenAI",
    }
    
    all_ok = True
    for module, name in try_imports.items(): 
        try:
            __import__(module)
            logger.info(f"  {name} ({module}) - OK")
        except ImportError as e:
            logger.error(f"  {name} ({module}): {e}")
            all_ok = False
    
    return all_ok


def test_file_structure() -> bool:
    """Valida a estrutura de diretórios e arquivos"""
    logger.info("\nValidando estrutura de arquivos...")
    
    required_files = [
        "src/generate_dataset.py",
        "src/finetune_llama.py",
        "src/inference.py",
        "src/config.py",
        "src/finetune_simple.py",
        "setup.py",
        "main.py",
        "requirements.txt",
        ".env.example",
        "README.md",
    ]
    
    required_dirs = [
        "src",
        "data",
        "models",
    ]
    
    all_ok = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            logger.info(f"  {file_path} - encontrado")
        else:
            logger.error(f"  {file_path} - não encontrado")
            all_ok = False
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.is_dir():
            logger.info(f"  {dir_path}/ - encontrado")
        else:
            logger.error(f"  {dir_path}/ - não encontrado")
            all_ok = False
    
    return all_ok


def test_code_syntax() -> bool:
    """Valida a sintaxe Python de todos os arquivos"""
    logger.info("\nValidando sintaxe Python...")
    
    python_files = [
        "src/generate_dataset.py",
        "src/finetune_llama.py",
        "src/finetune_simple.py",
        "src/inference.py",
        "src/config.py",
        "setup.py",
        "main.py",
    ]
    
    all_ok = True
    for file_path in python_files:
        try:
            with open(file_path) as f:
                compile(f.read(), file_path, 'exec')
            logger.info(f"  {file_path} - sintaxe OK")
        except SyntaxError as e:
            logger.error(f"  Erro de sintaxe em {file_path}: {e}")
            all_ok = False
    
    return all_ok


def test_required_configs() -> bool:
    """Valida as configurações obrigatórias de LoRA e QLoRA"""
    logger.info("\nValidando configurações obrigatórias...")
    
    configs = {
        "LoRA Rank (r=64)": ("src/finetune_simple.py", "r=64"),
        "LoRA Alpha (alpha=16)": ("src/finetune_simple.py", "lora_alpha=16"),
        "LoRA Dropout (0.1)": ("src/finetune_simple.py", "lora_dropout=0.1"),
        "QLoRA nf4": ("src/finetune_simple.py", 'bnb_4bit_quant_type="nf4"'),
        "QLoRA float16": ("src/finetune_simple.py", "bnb_4bit_compute_dtype=torch.float16"),
        "Optimizer paged_adamw_32bit": ("src/finetune_simple.py", "paged_adamw_32bit"),
        "Cosine scheduler": ("src/finetune_simple.py", "cosine"),
        "Warmup 0.03": ("src/finetune_simple.py", "warmup_ratio=0.03"),
        "CAUSAL_LM": ("src/finetune_simple.py", "CAUSAL_LM"),
    }
    
    all_ok = True
    for config_name, (file_path, search_term) in configs.items():
        try:
            with open(file_path) as f:
                content = f.read()
                if search_term in content:
                    logger.info(f"  {config_name} - encontrado")
                else:
                    logger.error(f"  {config_name} - não encontrado")
                    all_ok = False
        except Exception as e:
            logger.error(f"  Erro ao verificar {config_name}: {e}")
            all_ok = False
    
    return all_ok


def test_documentation() -> bool:
    """Valida se a documentação está presente"""
    logger.info("\nValidando documentação...")
    
    required_sections = [
        "## Objetivo",
        "## Estrutura",
        "## Passo",
        "## Conformidade",
    ]
    
    try:
        with open("README.md") as f:
            content = f.read()
            found = sum(1 for section in required_sections if section in content)
            
            if found >= 3:
                logger.info(f"  README.md validado ({found}/{len(required_sections)} seções)")
                return True
            else:
                logger.warning(f"  README.md: {found}/{len(required_sections)} seções encontradas")
                return False
    except Exception as e:
        logger.error(f"  Erro ao ler README.md: {e}")
        return False


def test_env_configuration() -> bool:
    """Valida a configuração do arquivo .env"""
    logger.info("\nValidando configuração .env...")
    
    env_example_exists = Path(".env.example").exists()
    env_exists = Path(".env").exists()
    
    if env_example_exists:
        logger.info("  .env.example - encontrado")
    else:
        logger.error("  .env.example - não encontrado")
        return False
    
    if env_exists:
        logger.info("  .env - encontrado (configurado)")
    else:
        logger.info("  Nota: .env não encontrado (copie .env.example)")
    
    return True


def main():
    """Executa todos os testes de validação"""
    logger.info("\n" + "="*60)
    logger.info("SUITE DE TESTES - VALIDAÇÃO DO PIPELINE")
    logger.info("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Estrutura de Arquivos", test_file_structure),
        ("Sintaxe Python", test_code_syntax),
        ("Configurações Obrigatórias", test_required_configs),
        ("Documentação", test_documentation),
        ("Configuração .env", test_env_configuration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Erro inesperado em {test_name}: {e}")
            results[test_name] = False
    
    # Resumo
    logger.info("\n" + "="*60)
    logger.info("RESUMO DOS TESTES")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "OK" if result else "ERRO"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} testes passaram")
    
    if passed == total:
        logger.info("\nTodos os testes passaram!")
        logger.info("O pipeline está pronto para submissão.\n")
        return 0
    else:
        logger.warning(f"\n{total - passed} teste(s) falharam.")
        logger.warning("Verifique os erros acima.\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
