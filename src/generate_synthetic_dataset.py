"""
Gerador de Dataset Sintético Local
Cria dados de treinamento sem depender de APIs externas
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dados sintéticos para assistência técnica
INSTRUCTION_TEMPLATES = [
    "Como resolver o problema: {}?",
    "Qual é a solução para {}?",
    "Explique como consertar {}",
    "Estou enfrentando {} - o que faço?",
    "Procedimento para resolver {}",
    "Dicas para lidar com {}",
    "Como diagnosticar {} em computadores?",
    "Guia rápido para {} no Windows",
]

PROBLEMS = [
    "tela azul da morte (BSOD)",
    "lentidão do sistema",
    "erro ao iniciar Windows",
    "disco rígido cheio",
    "memória RAM insuficiente",
    "driver desatualizado",
    "problema de conectividade de rede",
    "arquivo corrompido",
    "malware ou vírus",
    "problema de bateria no laptop",
    "superaquecimento do processador",
    "teclado não responde",
    "mouse travado",
    "som não funciona",
    "câmera não funciona",
    "impressora não conecta",
    "WiFi desconecta constantemente",
    "erro de atualização do Windows",
    "arquivo não abre",
    "programa congela regularmente",
]

SOLUTIONS = {
    "tela azul da morte (BSOD)": """Passos para resolver:
1. Reinicie o computador em Modo Seguro (F8 ou Shift+F8)
2. Verifique erros de disco: chkdsk /f
3. Atualize drivers de hardware
4. Desinstale programas recentemente instalados
5. Verifique temperatura do processador
6. Se persistir, considere recuperação do sistema ou reinstalação do Windows
Causa comum: driver incompatível ou falha de hardware""",

    "lentidão do sistema": """Soluções eficazes:
1. Limpe arquivos temporários (Limpeza de Disco)
2. Desative programas de inicialização (msconfig)
3. Aumente memória virtual ou RAM física
4. Desfragmente o disco (SSD: otimize, HDD: desfragmente)
5. Escaneie antivírus completo
6. Atualize drivers gráficos
7. Feche abas do navegador desnecessárias
Defragmente regularmente para melhor desempenho""",

    "erro ao iniciar Windows": """Procedimento de recuperação:
1. Tente Inicialização Segura (F8)
2. Use Reparar seu computador no DVD de instalação
3. Execute chkdsk /r /f
4. Restaure sistema para ponto anterior
5. Tente modo de segurança com prompt de comando
6. Verifique conectores internos (RAM, HDD)
7. Se tudo falhar, reinstale Windows
Causa: arquivo de boot corrompido ou falha de hardware""",

    "disco rígido cheio": """Libere espaço imediatamente:
1. Verifique C:\\Users\\[seu_usuário]\\AppData\\Local\\Temp
2. Limpe Lixeira
3. Desinstale programas não utilizados
4. Comprima arquivos antigos
5. Mova arquivos para HDD externo ou nuvem
6. Limpe cache de navegador
7. Aumente partição ou adicione segundo disco
Ideal: manter 15-20% livres no disco""",

    "memória RAM insuficiente": """Solução definitiva:
1. Verifique limite atual: Gerenciador de Tarefas > Desempenho
2. Feche programas desnecessários
3. Aumente memória virtual (Win+Pause > Avançado > Desempenho > Avançado)
4. Compre módulo RAM adicional compatível (DDR3/DDR4/DDR5)
5. Instale RAM nova em slots vazios
6. Atualize BIOS se necessário
Recomendado: mínimo 8GB (ideal 16GB+ para desenvolvimento)""",

    "driver desatualizado": """Atualização de drivers:
1. Identifique hardware: Gerenciador de Dispositivos
2. Clique direito > Atualizar driver
3. Pesquise automaticamente online
4. Se falhar, baixe do site do fabricante
5. Desinstale e reinstale se necessário
6. Reinicie após atualizar
7. Para GPU: NVIDIA GeForce Experience ou AMD Radeon Software
Atualize regularmente para melhor compatibilidade e segurança""",

    "problema de conectividade de rede": """Diagnóstico e resolução:
1. Verifique cabo Ethernet ou WiFi ativo
2. Reinicie roteador por 30 segundos
3. Execute ipconfig /all no cmd
4. Ping 8.8.8.8 para testar conectividade
5. Atualize driver de rede
6. Desative firewall temporariamente (teste)
7. Redefina TCP/IP: netsh int ip reset resetlog.txt
Se WiFi: tente esquecer e reconectar à rede""",

    "arquivo corrompido": """Recuperação de arquivo:
1. Verificar integridade: abra em programa diferente
2. Use software de recuperação (Recuva, EaseUS)
3. Acesse versão anterior: Propriedades > Versão Anterior
4. Restaure do backup se disponível
5. Tente abrir em Microsoft Office Online (nuvem)
6. Para ZIP corrupto: use WinRAR em modo de reparo
7. Último recurso: contate suporte de dados se crítico
Sempre manter backups de dados importantes""",

    "malware ou vírus": """Remoção completa:
1. Inicie em Modo Seguro com Rede
2. Escaneie com Windows Defender completo
3. Instale e execute Malwarebytes (versão gratuita)
4. Execute HitmanPro para detecção adicional
5. Limpe registro com CCleaner (cuidado!)
6. Desinstale programas suspeitos
7. Restaure sistema se persistir
Prevenção: manter antivírus atualizado e evitar downloads suspeitos""",

    "problema de bateria no laptop": """Diagnóstico de bateria:
1. Relatório de bateria: powercfg /batteryreport
2. Verifique saúde em Gerenciador de Tarefas
3. Limpeza de contatos da bateria com isopropanol
4. Calibre bateria: carga 100%, descarregue até 5%, carregue 100%
5. Desative RGB, ajuste brilho, reduza frequência CPU
6. Se <80% de saúde: substitua bateria nova
Duração esperada: 300-500 ciclos de carga completos""",

    "superaquecimento do processador": """Arrefecimento do sistema:
1. Monitore temperatura: HWiNFO, MSI Afterburner
2. Limpe poeira com ar comprimido (desligado)
3. Reaplique pasta térmica CPU (cada 2-3 anos)
4. Verificar ventiladores (RPM no BIOS)
5. Melhor airflow: monitor ventiladores traseiros/inferiores
6. Considere cooler melhor se >85°C em carga
Temperatura segura: <60°C ociosa, <80°C em carga máxima""",

    "teclado não responde": """Solução rápida:
1. Reconecte USB
2. Tente em porta USB diferente
3. Reinicie computador
4. Atualize driver de teclado em Gerenciador de Dispositivos
5. Desinstale e reinstale driver
6. Teste em modo seguro (excluir software)
7. Se wireless: troque bateria ou reconecte
Se nada funciona: teste com teclado com fio externo""",

    "mouse travado": """Diagnóstico e correção:
1. Limpe sensor óptico com pano seco
2. Limpe pad mousepad ou superfície
3. Se wireless: sincronize receptor/mouse (botão dedicado)
4. Atualize driver de mouse através Gerenciador de Dispositivos
5. Tente porta USB diferente
6. Desabilite eficiência energética USB no BIOS
7. Considere comprar mouse novo se mecânico desgastado
Tempo esperado: mouse óptico dura ~3-5 anos""",

    "som não funciona": """Recuperação de áudio:
1. Verifique volume: clique ícone som barra tarefas
2. Tente desconectar/conectar fone
3. Abra mixer de volume (Alt+Ctrl+F6 some builds)
4. Atualize driver áudio: Gerenciador de Dispositivos
5. Verifique BIOS se áudio integrado habilitado
6. Desinstale completamente driver, reinstale
7. Teste com fone diferente para isolar hardware
Se tela azul: driver desatualizado - atualizar imediatamente""",

    "câmera não funciona": """Solução para câmera:
1. Verifique permissões de câmera: Configurações > Privacidade
2. Atualize driver de câmera/chipset
3. Reconecte USB (se câmera externa)
4. Tente app diferente (Câmera nativa Windows)
5. Desabilite efeitos gráficos em VLC/Discord
6. Verifique se câmera não está física encoberta
7. Restaure webcam em BIOS se integrada
Aplicativos diferentes têm suporte variado de drivers""",

    "impressora não conecta": """Instalação e conexão:
1. Verifique cabo USB ou WiFi conectado
2. Reinicie impressora por 30 segundos
3. Vá para Configurações > Dispositivos > Impressoras
4. Clique Adicionar impressora
5. Baixe driver do site do fabricante
6. Para rede: configure IP estático na impressora
7. Teste papel e tinta/toner suficiente
Dica: mantenha driver atualizado para soluções de impressão""",

    "WiFi desconecta constantemente": """Estabilização de conexão:
1. Altere canal roteador (1, 6 ou 11 em 2.4GHz)
2. Afaste roteador de micro-ondas/bluetooth
3. Atualize firmware do roteador no painel admin
4. Reduza distância do roteador ou reposicione
5. Atualize driver de WiFi no computador
6. Configure DNS fixo (8.8.8.8 ou 1.1.1.1)
7. Se 5GHz disponível, teste em vez de 2.4GHz
Desconexões frequentes: geralmente interferência de RF""",

    "erro de atualização do Windows": """Correção de atualizações:
1. Reinstale atualizações: Configurações > Atualização > Verificar
2. Execute Windows Update Troubleshooter (Configurações)
3. Limpe pasta SoftwareDistribution: net stop wuauserv
4. Limpe: C:\\Windows\\SoftwareDistribution\\Download
5. Reinicie serviço: net start wuauserv
6. Se falhar: use Media Creation Tool para reparo
7. Aumente espaço livre em disco se <15% livre
Erro 0x800... : geralmente arquivo danificado ou espaço insuficiente""",

    "arquivo não abre": """Resolução para arquivo:
1. Verifique extensão correta (.txt, .pdf, etc)
2. Tente abrir com programa diferente: Abrir Com
3. Verify arquivo não está corrompido
4. Instale programa apropriado (ex: Adobe Reader para PDF)
5. Verifique permissões: Propriedades > Segurança
6. Tente restaurar versão anterior do arquivo
7. Use ferramentas online se programa não disponível
Arquivo não encontrado: verifique caminho completo""",

    "programa congela regularmente": """Diagnóstico de congelamento:
1. Monitore recursos em tempo real: Gerenciador de Tarefas
2. Verifique se disco está 100% em I/O
3. Atualize programa para versão compatível
4. Desinstale e reinstale programa
5. Verifique se RAM insuficiente (swap em disco lento)
6. Considere upgrade de SSD
7. Procure patches/hotfixes de bug programa
Se apenas programa: considere alternativa mais estável""",
}

def generate_synthetic_dataset(num_samples=50, output_dir="data"):
    """Generate synthetic dataset locally without API"""
    
    logger.info(f"Gerando {num_samples} pares sintéticos...")
    dataset = []
    
    # Create more diverse pairs by systematic combination
    problem_cycle = 0
    template_cycle = 0
    
    for i in range(num_samples):
        problem = PROBLEMS[problem_cycle % len(PROBLEMS)]
        template = INSTRUCTION_TEMPLATES[template_cycle % len(INSTRUCTION_TEMPLATES)]
        
        instruction = template.format(problem)
        output = SOLUTIONS.get(problem, f"Solução para resolver {problem}. Por favor, consulte documentação oficial ou técnico especializado.")
        
        dataset.append({
            "instruction": instruction,
            "output": output
        })
        
        problem_cycle += 1
        if problem_cycle % 3 == 0:
            template_cycle += 1
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into train (90%) and test (10%)
    train_size = int(len(dataset) * 0.9)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    # Write JSONL files
    train_path = os.path.join(output_dir, "train_dataset.jsonl")
    test_path = os.path.join(output_dir, "test_dataset.jsonl")
    full_path = os.path.join(output_dir, "full_dataset.jsonl")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"✅ Train dataset salvo: {train_path} ({len(train_dataset)} pares)")
    
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"✅ Test dataset salvo: {test_path} ({len(test_dataset)} pares)")
    
    with open(full_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"✅ Full dataset salvo: {full_path} ({len(dataset)} pares)")
    
    return dataset

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("GERADOR DE DATASET SINTÉTICO")
    logger.info("=" * 60)
    
    dataset = generate_synthetic_dataset(num_samples=50)
    
    logger.info("\n" + "=" * 60)
    logger.info("SUCESSO! Dataset gerado com 50 pares de dados")
    logger.info("=" * 60)
    logger.info(f"Total de problemas únicos: {len(PROBLEMS)}")
    logger.info(f"Total de templates: {len(INSTRUCTION_TEMPLATES)}")
    logger.info(f"Pares gerados: {len(dataset)}")
