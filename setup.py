"""
Script de setup e validação do ambiente
Verifica dependências, configurações e GPU antes de executar o pipeline
"""

import os
import sys
import torch
import subprocess
from pathlib import Path
from typing import Tuple, List  

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version() -> bool:
    """Verifica se está usando Python 3.9+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.error(f"❌ Python 3.9+ requerido (encontrado: {version.major}.{version.minor})")
        return False


def check_gpu() -> Tuple[bool, str]:
    """Verifica disponibilidade de GPU CUDA"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU detectada: {device_name}")
        logger.info(f"   Memória: {memory_gb:.2f} GB")
        return True, device_name
    else:
        logger.warning("⚠️  GPU não detectada. CPU será usada (muito lento).")
        return False, "CPU"


def check_cuda_toolkit() -> bool:
    """Verifica CUDA toolkit"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ CUDA Toolkit detectado")
            return True
        else:
            logger.warning("⚠️  CUDA Toolkit não encontrado")
            return False
    except FileNotFoundError:
        logger.warning("⚠️  nvidia-smi não encontrado (CUDA pode estar instalado)")
        return False


def check_dependencies() -> Tuple[bool, List[str]]:
    """Verifica dependências Python críticas"""
    missing = []
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "trl",
        "bitsandbytes",
        "openai",
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} não instalado")
            missing.append(package)
    
    if missing:
        logger.error(f"\n❌ Pacotes faltando: {', '.join(missing)}")
        logger.info(f"Instale com: pip install {' '.join(missing)}")
        return False, missing
    
    logger.info("✅ Todas as dependências instaladas")
    return True, []


def check_environment_file() -> bool:
    """Verifica arquivo .env"""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        logger.info("✅ Arquivo .env encontrado")
        
        # Verificar se OPENAI_API_KEY está configurada
        with open(env_path) as f:
            content = f.read()
            if "OPENAI_API_KEY" in content and "seu_openai_api_key" not in content.lower():
                logger.info("✅ OPENAI_API_KEY configurada")
                return True
            else:
                logger.warning("⚠️  OPENAI_API_KEY não configurada no .env")
                return False
    else:
        logger.warning("⚠️  Arquivo .env não encontrado")
        if env_example_path.exists():
            logger.info(f"   Use: cp .env.example .env")
        return False


def check_data_directory() -> bool:
    """Verifica estrutura de diretórios"""
    required_dirs = ["data", "src", "models"]
    
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            logger.info(f"✅ Diretório {dir_name}/ existe")
        else:
            logger.warning(f"⚠️  Diretório {dir_name}/ não encontrado")
    
    return True


def check_model_access() -> bool:
    """Testa acesso ao modelo Llama 2"""
    try:
        from transformers import AutoConfig
        logger.info("⏳ Testando acesso ao modelo Llama 2...")
        # Este teste é rápido (apenas baixa config)
        config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
        logger.info("✅ Acesso ao modelo Llama 2 funcionando")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Não foi possível acessar Llama 2: {e}")
        logger.info("   Pode ser necessário: huggingface-cli login")
        return False


def print_system_info():
    """Exibe informações do sistema"""
    logger.info("\n" + "="*60)
    logger.info("INFORMAÇÕES DO SISTEMA")
    logger.info("="*60)
    
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CPU: {torch.get_num_threads()} threads")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"cuDNN: {torch.backends.cudnn.version()}")
    
    logger.info("="*60 + "\n")


def main():
    """Executa todas as verificações"""
    logger.info("\n" + "="*60)
    logger.info("VERIFICAÇÃO DO AMBIENTE")
    logger.info("="*60 + "\n")
    
    print_system_info()
    
    # Verificações
    checks = [
        ("Python 3.9+", check_python_version),
        ("GPU CUDA", check_gpu),
        ("CUDA Toolkit", check_cuda_toolkit),
        ("Dependências Python", check_dependencies),
        ("Arquivo .env", check_environment_file),
        ("Estrutura de diretórios", check_data_directory),
        ("Acesso ao modelo Llama 2", check_model_access),
    ]
    
    results = {}
    for name, check_func in checks:
        logger.info(f"\n🔍 Verificando: {name}")
        logger.info("-" * 60)
        try:
            if name == "GPU CUDA":
                success, device = check_gpu()
                results[name] = success
            elif name == "Dependências Python":
                success, missing = check_dependencies()
                results[name] = success
            else:
                results[name] = check_func()
        except Exception as e:
            logger.error(f"Erro ao verificar {name}: {e}")
            results[name] = False
    
    # Resumo
    logger.info("\n" + "="*60)
    logger.info("RESUMO")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {name}")
    
    logger.info(f"\nTotal: {passed}/{total} verificações passadas")
    
    if passed == total:
        logger.info("\n✅ Ambiente pronto! Execute:")
        logger.info("   python src/generate_dataset.py  # Gerar dados")
        logger.info("   python src/finetune_llama.py    # Fine-tuning")
        return 0
    elif passed >= total - 2:
        logger.warning("\n⚠️  Ambiente parcialmente configurado.")
        logger.warning("   Verifique os erros acima antes de executar.")
        return 1
    else:
        logger.error("\n❌ Ambiente não pronto para execução.")
        logger.error("   Corrija os erros acima antes de prosseguir.")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
