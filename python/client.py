import requests
import random
import time
import threading
import os 
import datetime

# URLs de reconstrução
URL_PYTHON_SERVER = "http://localhost:5000/interpretedServer/reconstruct"
URL_JAVA_SERVER = "http://localhost:8080/interpretedServer/reconstruct" 

# Um Lock global para proteger a escrita nos arquivos de log
file_lock = threading.Lock()

MODELS = ['H_60x60.csv','H_30x30.csv']
SIGNAL60 = ['sinal_1_60x60.csv',
            'sinal_2_60x60.csv',
            'sinal_3_60x60.csv']
SIGNAL30 = ['sinal_1_30x30.csv',
            'sinal_2_30x30.csv',
            'sinal_3_30x30.csv']
ALGORITHM = ['CGNE','CGNR']

def make_request(target_url, server_tag, payload, model, signal, algorithm):
    """
    Função 'worker' que executa a requisição e processa a resposta.
    Esta função é chamada em uma thread separada para cada servidor.
    """
    try:
        print(f"[REQUISIÇÃO] Enviando para {server_tag.upper()}...")
        start_req_time = time.time()
        resp = requests.post(target_url, json=payload, timeout=60.0)
        end_req_time = time.time()
        
        req_duration = end_req_time - start_req_time

        if resp.status_code == 200:
            image_name = f"img_{algorithm}_{signal}_{server_tag}.png"
            output_dir = "Reconstructed"
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, image_name)
            
            with open(image_path, 'wb') as f:
                f.write(resp.content)
            
            # Extrai os headers da resposta do servidor
            iteractions = resp.headers.get('X-Iteracoes', '0')
            exec_time = resp.headers.get('X-Tempo', '0') 
            alg = resp.headers.get('X-Algoritmo', 'unknown')
            start = resp.headers.get('X-Inicio', '')
            finish = resp.headers.get('X-Fim', '')
            size = resp.headers.get('X-Tamanho', '')
            uso_cpu = resp.headers.get('X-Cpu','')
            uso_mem = resp.headers.get('X-Mem','')

            # --- Escrita segura em arquivo ---
            with file_lock:
                with open('relatorio_imagens.txt', 'a') as f:
                    f.write(
                        f"{image_name} - Arquivo:  {alg}, "
                        f"Inicio: {start}, Fim: {finish}, Tamanho: {size}, "
                        f"Iterações: {iteractions}, Tempo Servidor: {exec_time} s, "
                        f"Tempo Req Total: {req_duration:.4f} s, Modelo: {model}, Sinal: {signal}\n"
                    )
            print(f"[SUCESSO] {server_tag.upper()} - Imagem salva: {image_name} ({iteractions} iterações, {exec_time}s)")

            with file_lock:
                with open('relatorio_desempenho.txt', 'a') as f:
                    f.write(
                        f"[{finish}] Servidor: {server_tag.upper()}, CPU: {uso_cpu}%, Memória: {uso_mem}%\n"
                    )
            print(f"[DESEMPENHO] {server_tag.upper()} - CPU: {uso_cpu}%, Memória: {uso_mem}%")
        else:
            print(f"[ERRO] {server_tag.upper()} - Resposta: {resp.status_code} - {resp.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"[ERRO] {server_tag.upper()} - Falha na comunicação: {e}")

def send_signal():
    """
    Função 'master' que prepara a carga e dispara as threads 
    para enviar simultaneamente aos servidores Python e Java.
    """
    model = 'H_30x30.csv'# random.choice(MODELS)

    if model == 'H_60x60.csv':
        signal = random.choice(SIGNAL60)
    else:
        signal = 'sinal_1_30x30.csv'#random.choice(SIGNAL30)

    algorithm = random.choice(ALGORITHM)

    print(f"\n[ENVIO SIMULTÂNEO] Modelo: {model}, Sinal: {signal}, Alg: {algorithm}") 

    payload = {
        'algoritmo': algorithm,
        'modelo': model,
        'sinal': signal
    }

    # Cria uma thread para cada servidor, passando os mesmos argumentos
    thread_python = threading.Thread(target=make_request, 
                                     args=(URL_PYTHON_SERVER, "python", payload, model, signal, algorithm))
    
    thread_java = threading.Thread(target=make_request, 
                                   args=(URL_JAVA_SERVER, "java", payload, model, signal, algorithm))

    # Inicia ambas as threads (quase) ao mesmo tempo
    thread_python.start()
    thread_java.start()

    # Aguarda que ambas as requisições terminem antes de continuar
    thread_python.join()
    thread_java.join()
    print(f"[ENVIO CONCLUÍDO] {model} / {signal}")


# ===================================================================
# NENHUMA ALTERAÇÃO NECESSÁRIA ABAIXO
# As funções de simulação (executar_cliente, funcao_thread_cliente, 
# simula_clientes) funcionam como estão. Elas chamam `send_signal`,
# que agora faz o trabalho duplicado.
# ===================================================================

def executar_cliente(num_sinais=5):
    print("=== CLIENTE DE RECONSTRUÇÃO DE IMAGENS DE ULTRASSOM ===")
    print(f"Enviando {num_sinais} sinais simultaneamente para Java e Python...")
    
    for arquivo in ['relatorio_imagens.txt', 'relatorio_desempenho.txt']:
        if os.path.exists(arquivo):
            os.remove(arquivo)
    
    with open('relatorio_imagens.txt', 'w') as f:
        f.write("=== RELATÓRIO DE IMAGENS RECONSTRUÍDAS ===\n")
        f.write(f"Gerado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    with open('relatorio_desempenho.txt', 'w') as f:
        f.write("=== RELATÓRIO DE DESEMPENHO DOS SERVIDORES ===\n")
        f.write(f"Gerado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for i in range(num_sinais):
        print(f"\n--- Disparando sinal {i+1}/{num_sinais} ---")
        
        send_signal()
        
        if i < num_sinais - 1:  
            intervalo = random.uniform(1, 3)
            print(f"[INTERVALO] Aguardando {intervalo:.1f}s...")
            time.sleep(intervalo)

    print("\n=== EXECUÇÃO CONCLUÍDA ===")
    print("Arquivos gerados:")
    print("- relatorio_imagens.txt: Relatório das imagens reconstruídas (ambos servidores)")
    print("- relatorio_desempenho.txt: Relatório de desempenho dos servidores (ambos servidores)")
    print("- img_*.png: Imagens reconstruídas (tags _python ou _java)")

def funcao_thread_sinal():
    send_signal()
    
def funcao_thread_cliente():
    quantidade_sinais = 2 #random.randint(1,3)
    threads_sinal = []

    for j in range(quantidade_sinais):
        thread_sinal = threading.Thread(target=funcao_thread_sinal)
        thread_sinal.start()
        threads_sinal.append(thread_sinal)
        time.sleep(random.randint(1,10))

    for t in threads_sinal:
        t.join()

def simula_clientes():
    quantidade_clientes = 1 #random.randint(2,3)
    threads_cliente = []
    for i in range(quantidade_clientes):
        thread_cliente = threading.Thread(target=funcao_thread_cliente)
        thread_cliente.start()
        threads_cliente.append(thread_cliente)

    print(f"[MAIN_CLIENT] {len(threads_cliente)} clientes iniciados. Aguardando conclusão...")
    for t in threads_cliente:
        t.join() 
    print("[MAIN_CLIENT] Todos os clientes concluíram suas tarefas.")