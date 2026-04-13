"""
Inference script with fine-tuned model
Demonstrates how to load LoRA adapter and make predictions
"""

import torch
import os
from pathlib import Path
from typing import Optional
import logging

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LLMInference:
    """Classe para inferência com modelo fine-tuned"""
    
    def __init__(
        self,
        adapter_path: str,
        device: str = "auto"
    ):
        """
        Inicializa o modelo para inferência.
        
        Args:
            adapter_path: Caminho para o adaptador LoRA salvo
            device: Dispositivo para carregar o modelo (auto, cuda, cpu)
        """
        self.adapter_path = adapter_path
        self.device = device
        
        logger.info(f"Carregando modelo do adaptador: {adapter_path}")
        
        # Carregar modelo com adaptador
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            device_map=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            is_trainable=False,
        )
        
        # Carregar tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        logger.info("Modelo carregado com sucesso")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Gera resposta para um prompt.
        
        Args:
            prompt: Texto de entrada
            max_new_tokens: Número máximo de tokens a gerar
            temperature: Temperatura (controla criatividade)
            top_p: Top-p para nucleus sampling
        
        Returns:
            Texto gerado
        """
        # Tokenizar entrada
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)
        
        logger.info(f"Processando: {prompt[:100]}...")
        
        # Gerar saída
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decodificar resultado
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remover o prompt da resposta
        response = response[len(prompt):].strip()
        
        return response
    
    def generate_batch(
        self,
        prompts: list,
        max_new_tokens: int = 256,
    ) -> list:
        """
        Gera respostas para múltiplos prompts.
        
        Args:
            prompts: Lista de textos de entrada
            max_new_tokens: Número máximo de tokens
        
        Returns:
            Lista de respostas
        """
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"  Processando {i+1}/{len(prompts)}...")
            response = self.generate(prompt, max_new_tokens)
            responses.append(response)
        
        return responses


def interactive_mode():
    """
    Modo interativo para fazer perguntas ao modelo
    """
    # Encontrar caminho do adaptador mais recente
    models_dir = Path("models")
    adapter_dirs = sorted(
        [d for d in models_dir.glob("llama2-finetuned*")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not adapter_dirs:
        logger.error("Nenhum modelo treinado encontrado em models/")
        logger.info("Execute primeiro: python src/finetune_llama.py")
        return
    
    adapter_path = adapter_dirs[0] / "adapter_model"
    
    logger.info(f"Usando modelo: {adapter_path}")
    logger.info("="*60)
    
    # Carregar modelo
    inference = LLMInference(str(adapter_path))
    
    logger.info("\nModo Interativo - Digite 'sair' para encerrar\n")
    logger.info("="*60)
    
    while True:
        try:
            prompt = input("\nSua pergunta: ").strip()
            
            if prompt.lower() in ["sair", "exit", "quit"]:
                logger.info("Encerrando...")
                break
            
            if not prompt:
                logger.warning("Por favor, digite uma pergunta válida")
                continue
            
            response = inference.generate(prompt)
            
            print(f"\nResposta:\n{response}")
            print("-"*60)
        
        except KeyboardInterrupt:
            logger.info("\nEncerrando...")
            break
        except Exception as e:
            logger.error(f"Erro: {e}")


def batch_inference_example():
    """
    Exemplo de inferência em lote
    """
    # Encontrar caminho do adaptador
    models_dir = Path("models")
    adapter_dirs = sorted(
        [d for d in models_dir.glob("llama2-finetuned*")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not adapter_dirs:
        logger.error("Nenhum modelo treinado encontrado")
        return
    
    adapter_path = adapter_dirs[0] / "adapter_model"
    
    # Exemplos de prompts
    prompts = [
        "Como diagnosticar problemas de conectividade Wi-Fi?",
        "Qual é a melhor forma de manter o computador seguro?",
        "Como melhorar a performance do SSD?",
    ]
    
    logger.info("Iniciando inferência em lote...\n")
    
    # Carregar modelo
    inference = LLMInference(str(adapter_path))
    
    # Gerar respostas
    logger.info("Processando prompts...")
    responses = inference.generate_batch(prompts)
    
    # Exibir resultados
    logger.info("\n" + "="*60)
    logger.info("RESULTADOS")
    logger.info("="*60 + "\n")
    
    for prompt, response in zip(prompts, responses):
        print(f"❓ Pergunta: {prompt}")
        print(f"Resposta: {response[:200]}...")
        print("-"*60)


def main():
    """Função principal"""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        batch_inference_example()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
