import requests
import random
import time
import threading
import os            
import datetime

URL_JAVA_SERVER = "http://localhost:8080/compiledServe/reconstruct"
URL_PYTHON_SERVER = "http://localhost:5000/interpretedServer/reconstruct"

MODELS = ['H_60x60.csv','H_30x30.csv']
SIGNAL60 = ['sinal_1_60x60.csv',
            'sinal_2_60x60.csv',
            'sinal_3_60x60.csv']
SIGNAL30 = ['sinal_1_30x30.csv',
            'sinal_2_30x30.csv',
            'sinal_3_30x30.csv']
ALGORITHM = ['CGNE','CGNR']

def send_signal():
    model = random.choice(MODELS)

    if model == 'H_60x60.csv':
        signal = random.choice(SIGNAL60)
    else:
        signal = random.choice(SIGNAL30)

    algorithm = random.choice(ALGORITHM)

    print(f"[LOG] Enviando sinal -> Modelo = {model}, Sinal = {signal}")    

    payload = {
        'algoritmo': algorithm,
        'modelo': model,
        'sinal': signal
    }

    try:
        resp = requests.post(URL_PYTHON_SERVER,json=payload)
        if resp.status_code == 200:
            image_name = f"img_{algorithm}_{signal}_python.png"
            output_dir = "Reconstructed"
            os.makedirs(output_dir, exist_ok=True) # Cria o dir se não existir
            image_path = os.path.join(output_dir, image_name)
            with open(image_path, 'wb') as f:
                f.write(resp.content)
            iteractions = resp.headers.get('X-Iteracoes', '0')
            time = resp.headers.get('X-Tempo', '0')
            alg = resp.headers.get('X-Algoritmo', 'unknown')
            start = resp.headers.get('X-Inicio', '')
            finish = resp.headers.get('X-Fim', '')
            size = resp.headers.get('X-Tamanho', '')

            uso_cpu = resp.headers.get('X-Cpu','')
            uso_mem = resp.headers.get('X-Mem','')
            with open('relatorio_imagens.txt', 'a') as f:
                f.write(
                    f"{image_name} - Arquivo:  {alg}, "
                    f"Inicio: {start}, Fim: {finish}, Tamanho: {size}, "
                    f"Iterações: {iteractions}, Tempo: {time} s, Modelo: {model}, Sinal: {signal}\n"
                )
            print(f"[SUCESSO] Imagem salva: {image_name} ({iteractions} iterações, {time}s)")

            with open('relatorio_desempenho.txt', 'a') as f:
                f.write(
                    f"[{finish}] CPU: {uso_cpu}%, Memória: {uso_mem}%\n"
                )
            print(f"[DESEMPENHO] CPU: {uso_cpu}%, Memória: {uso_mem}%")
        else:
            print(f"[ERRO] Resposta do servidor: {resp.status_code} - {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"[ERRO] Falha na comunicação com servidor: {e}")

def executar_cliente(num_sinais=5):
    print("=== CLIENTE DE RECONSTRUÇÃO DE IMAGENS DE ULTRASSOM ===")
    print(f"Enviando {num_sinais} sinais com intervalos aleatórios...")
    
    for arquivo in ['relatorio_imagens.txt', 'relatorio_desempenho.txt']:
        if os.path.exists(arquivo):
            os.remove(arquivo)
    
    with open('relatorio_imagens.txt', 'w') as f:
        f.write("=== RELATÓRIO DE IMAGENS RECONSTRUÍDAS ===\n")
        f.write(f"Gerado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    with open('relatorio_desempenho.txt', 'w') as f:
        f.write("=== RELATÓRIO DE DESEMPENHO DO SERVIDOR ===\n")
        f.write(f"Gerado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for i in range(num_sinais):
        print(f"\n--- Enviando sinal {i+1}/{num_sinais} ---")
        
        send_signal()
        
        if i < num_sinais - 1:  
            intervalo = random.uniform(1, 3)
            print(f"[INTERVALO] Aguardando {intervalo:.1f}s...")
            time.sleep(intervalo)

    print("\n=== EXECUÇÃO CONCLUÍDA ===")
    print("Arquivos gerados:")
    print("- relatorio_imagens.txt: Relatório das imagens reconstruídas")
    print("- relatorio_desempenho.txt: Relatório de desempenho do servidor")
    print("- img_*.png: Imagens reconstruídas")

def funcao_thread_sinal():
    send_signal()
    
def funcao_thread_cliente():
    quantidade_sinais = random.randint(2,5)
    threads_sinal = []

    for j in range(quantidade_sinais):
        thread_sinal = threading.Thread(target=funcao_thread_sinal)
        thread_sinal.start()
        threads_sinal.append(thread_sinal)
        time.sleep(random.randint(1,10))

    for t in threads_sinal:
        t.join()

def simula_clientes():
    quantidade_clientes = 1
    threads_cliente = []
    for i in range(quantidade_clientes):
        thread_cliente = threading.Thread(target=funcao_thread_cliente)
        thread_cliente.start()
        threads_cliente.append(thread_cliente)

    print(f"[MAIN_CLIENT] {len(threads_cliente)} clientes iniciados. Aguardando conclusão...")
    for t in threads_cliente:
        t.join() 
    print("[MAIN_CLIENT] Todos os clientes concluíram suas tarefas.")