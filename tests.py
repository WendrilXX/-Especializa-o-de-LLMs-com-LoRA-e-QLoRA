"""
Script de Testes e Validação
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
    """Testa se todos os imports funcionam"""
    logger.info("\n🧪 Testando imports...")
    
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
            logger.info(f"  ✅ {name} ({module})")
        except ImportError as e:
            logger.error(f"  ❌ {name} ({module}): {e}")
            all_ok = False
    
    return all_ok


def test_file_structure() -> bool:
    """Testa se a estrutura de diretórios está ok"""
    logger.info("\n🧪 Testando estrutura de arquivos...")
    
    required_files = [
        "src/generate_dataset.py",
        "src/finetune_llama.py",
        "src/inference.py",
        "src/config.py",
        "setup.py",
        "main.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        "QUICKSTART.md",
        "TECHNICAL.md",
        "CHECKLIST.md",
    ]
    
    required_dirs = [
        "src",
        "data",
        "models",
    ]
    
    all_ok = True
    
    # Verificar arquivos
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            logger.info(f"  ✅ {file_path}")
        else:
            logger.error(f"  ❌ {file_path} não encontrado")
            all_ok = False
    
    # Verificar diretórios
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.is_dir():
            logger.info(f"  ✅ {dir_path}/")
        else:
            logger.error(f"  ❌ {dir_path}/ não encontrado")
            all_ok = False
    
    return all_ok


def test_code_syntax() -> bool:
    """Testa se o código Python é válido"""
    logger.info("\n🧪 Testando sintaxe Python...")
    
    python_files = [
        "src/generate_dataset.py",
        "src/finetune_llama.py",
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
            logger.info(f"  ✅ {file_path}")
        except SyntaxError as e:
            logger.error(f"  ❌ {file_path}: {e}")
            all_ok = False
    
    return all_ok


def test_required_configs() -> bool:
    """Testa se as configurações obrigatórias estão presentes"""
    logger.info("\n🧪 Testando configurações obrigatórias...")
    
    configs = {
        "LoRA Rank (r=64)": ("src/finetune_llama.py", "r=64"),
        "LoRA Alpha (alpha=16)": ("src/finetune_llama.py", "lora_alpha=16"),
        "LoRA Dropout (0.1)": ("src/finetune_llama.py", "lora_dropout=0.1"),
        "QLoRA nf4": ("src/finetune_llama.py", 'bnb_4bit_quant_type="nf4"'),
        "QLoRA float16": ("src/finetune_llama.py", "torch.float16"),
        "Otimizador paged": ("src/finetune_llama.py", "paged_adamw_32bit"),
        "Scheduler cosine": ("src/finetune_llama.py", 'lr_scheduler_type="cosine"'),
        "Warmup 0.03": ("src/finetune_llama.py", "warmup_ratio=0.03"),
        "CAUSAL_LM": ("src/finetune_llama.py", "CAUSAL_LM"),
    }
    
    all_ok = True
    for config_name, (file_path, search_term) in configs.items():
        try:
            with open(file_path) as f:
                content = f.read()
                if search_term in content:
                    logger.info(f"  ✅ {config_name}")
                else:
                    logger.error(f"  ❌ {config_name} não encontrado em {file_path}")
                    all_ok = False
        except Exception as e:
            logger.error(f"  ❌ Erro ao verificar {config_name}: {e}")
            all_ok = False
    
    return all_ok


def test_documentation() -> bool:
    """Testa se a documentação está completa"""
    logger.info("\n🧪 Testando documentação...")
    
    docs = {
        "README.md": ["## Objetivo", "## 🚀 Configuração", "## 📊 Passo"],
        "QUICKSTART.md": ["## 5 Passos", "python src/", "✅ Dataset"],
        "TECHNICAL.md": ["## Arquitetura", "## LoRA", "## Quantização"],
        "CHECKLIST.md": ["✅ Requisitos", "## 🔍 Como Verificar", "## 📁 Arquivos"],
    }
    
    all_ok = True
    for doc_file, required_sections in docs.items():
        try:
            with open(doc_file) as f:
                content = f.read()
                found = 0
                for section in required_sections:
                    if section in content:
                        found += 1
                
                if found == len(required_sections):
                    logger.info(f"  ✅ {doc_file} ({found} seções encontradas)")
                else:
                    logger.warning(f"  ⚠️  {doc_file} ({found}/{len(required_sections)} seções)")
        except Exception as e:
            logger.error(f"  ❌ Erro ao ler {doc_file}: {e}")
            all_ok = False
    
    return all_ok


def test_ai_attribution() -> bool:
    """Verifica se há atribuição de IA no README"""
    logger.info("\n🧪 Testando atribuição de IA...")
    
    try:
        with open("README.md") as f:
            content = f.read()
            if "IA" in content and "revista" in content.lower():
                logger.info("  ✅ Atribuição de IA presente no README")
                return True
            else:
                logger.warning("  ⚠️  Adicione atribuição de IA no README")
                return False
    except Exception as e:
        logger.error(f"  ❌ Erro ao verificar README: {e}")
        return False


def test_env_configuration() -> bool:
    """Verifica arquivo .env"""
    logger.info("\n🧪 Testando configuração .env...")
    
    env_exists = Path(".env").exists()
    env_example_exists = Path(".env.example").exists()
    
    if env_example_exists:
        logger.info("  ✅ .env.example encontrado")
    else:
        logger.error("  ❌ .env.example não encontrado")
        return False
    
    if env_exists:
        logger.info("  ✅ .env encontrado (configurado)")
        return True
    else:
        logger.warning("  ⚠️  .env não encontrado (copie .env.example)")
        return True  # Não é erro crítico


def main():
    """Executa todos os testes"""
    logger.info("\n" + "="*60)
    logger.info("🧪 SUITE DE TESTES - VALIDAÇÃO DO PIPELINE")
    logger.info("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Estrutura de Arquivos", test_file_structure),
        ("Sintaxe Python", test_code_syntax),
        ("Configurações Obrigatórias", test_required_configs),
        ("Documentação", test_documentation),
        ("Atribuição de IA", test_ai_attribution),
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
    logger.info("📊 RESUMO DOS TESTES")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} testes passaram")
    
    if passed == total:
        logger.info("\n✅ TODOS OS TESTES PASSARAM!")
        logger.info("O pipeline está pronto para submissão.\n")
        return 0
    else:
        logger.warning(f"\n⚠️  {total - passed} teste(s) falharam.")
        logger.warning("Verifique os erros acima.\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
