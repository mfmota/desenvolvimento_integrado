import requests
import random
import time
import threading
import os
import datetime
import numpy as np

URL_PYTHON_SERVER = "http://localhost:5000/interpretedServer/reconstruct"
URL_JAVA_SERVER = "http://localhost:8080/compiledServer/reconstruct"

file_lock = threading.Lock()

MODELS = ['H_60x60.csv', 'H_30x30.csv']
SIGNAL60 = ['sinal_1_60x60.csv', 'sinal_2_60x60.csv', 'sinal_3_60x60.csv']
SIGNAL30 = ['sinal_1_30x30.csv', 'sinal_2_30x30.csv', 'sinal_3_30x30.csv']
ALGORITHM = ['CGNE', 'CGNR']

def make_request(target_url, server_tag, sinal_bin, tamanho, model, signal, algorithm, gain):
    try:
        print(f"[REQUISIÇÃO] Enviando para {server_tag.upper()}...")

        headers = {
            "Content-Type": "application/octet-stream",
            "X-Modelo": model,
            "X-Alg": algorithm,
            "X-Tamanho": str(tamanho),
            "X-Ganho": str(gain),
        }

        start_req_time = time.time()
        resp = requests.post(target_url, data=sinal_bin, headers=headers)
        end_req_time = time.time()

        req_duration = end_req_time - start_req_time

        if resp.status_code == 200:
            image_name = f"img_{algorithm}_{signal}_{server_tag}.png"
            output_dir = "Reconstructed"
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, image_name)

            with open(image_path, 'wb') as f:
                f.write(resp.content)

            iteracoes = resp.headers.get('X-Iteracoes', '0')
            exec_time = resp.headers.get('X-Tempo', '0')
            alg = resp.headers.get('X-Algoritmo', 'unknown')
            start = resp.headers.get('X-Inicio', '')
            finish = resp.headers.get('X-Fim', '')
            size = resp.headers.get('X-Tamanho', '')
            uso_cpu = resp.headers.get('X-Cpu', '')
            uso_mem = resp.headers.get('X-Mem', '')

            with file_lock:
                with open('relatorio_imagens.txt', 'a') as f:
                    f.write(
                        f"{image_name} - Arquivo: {alg}, "
                        f"Inicio: {start}, Fim: {finish}, Tamanho: {size}, "
                        f"Iterações: {iteracoes}, Tempo Servidor: {exec_time} s, "
                        f"Tempo Req Total: {req_duration:.4f} s, Modelo: {model}, "
                        f"Sinal: {signal}, Ganho: {gain}\n"
                    )

            with file_lock:
                with open('relatorio_desempenho.txt', 'a') as f:
                    f.write(
                        f"[{finish}] Servidor: {server_tag.upper()}, CPU: {uso_cpu}%, Memória: {uso_mem}%\n"
                    )

            print(f"[SUCESSO] {server_tag.upper()} - Imagem salva.")

        else:
            print(f"[ERRO] {server_tag.upper()} - Resposta: {resp.status_code} - {resp.text}")

    except requests.exceptions.RequestException as e:
        print(f"[ERRO] {server_tag.upper()} - Falha na comunicação: {e}")


def send_signal():
    model = random.choice(MODELS)
    gain = "Dinâmico"

    if model == 'H_60x60.csv':
        filename = random.choice(SIGNAL60)
    else:
        filename = random.choice(SIGNAL30)

    signal_array = np.loadtxt(filename, delimiter=",").flatten()[:, np.newaxis]

    S = signal_array.shape[0]

    
    #l_indices = np.arange(1, S + 1, dtype=np.float64) 
    #gamma_vector = 100.0 + (1.0 / 20.0) * l_indices * np.sqrt(l_indices)
    #gamma_vector_reshaped = gamma_vector[:, np.newaxis]
    #signal_gain = signal_array * gamma_vector_reshaped

    #sinal_bin = signal_gain.astype(np.float32).tobytes()
    #tamanho = len(signal_gain)
    
    sinal_bin = signal_array.astype(np.float32).tobytes()
    tamanho = len(signal_array)

    print(f"\n[ENVIO] Modelo: {model}, Sinal: {filename}, Ganho: {gain}")

    for algorithm in ALGORITHM:
        thread_python = threading.Thread(
            target=make_request,
            args=(URL_PYTHON_SERVER, "python", sinal_bin, tamanho, model, filename, algorithm, gain)
        )

        thread_java = threading.Thread(
            target=make_request,
            args=(URL_JAVA_SERVER, "java", sinal_bin, tamanho, model, filename, algorithm, gain)
        )

        thread_python.start()
        thread_java.start()

        thread_python.join()
        thread_java.join()

    print(f"[ENVIO CONCLUÍDO] {filename} (com ganho γ={gain})")


def executar_cliente(num_sinais=5):
    print("=== CLIENTE INICIADO ===")

    for arq in ['relatorio_imagens.txt', 'relatorio_desempenho.txt']:
        if os.path.exists(arq):
            os.remove(arq)

    with open('relatorio_imagens.txt', 'w') as f:
        f.write("=== RELATÓRIO ===\n")
        f.write(f"{datetime.datetime.now()}\n\n")

    with open('relatorio_desempenho.txt', 'w') as f:
        f.write("=== DESEMPENHO ===\n")
        f.write(f"{datetime.datetime.now()}\n\n")

    for i in range(num_sinais):
        print(f"\n--- Enviando sinal {i+1}/{num_sinais} ---")
        send_signal()
        if i < num_sinais - 1:
            t = random.uniform(1, 3)
            print(f"[INTERVALO] Aguardando {t:.1f}s...")
            time.sleep(t)

    print("=== EXECUÇÃO FINALIZADA ===")