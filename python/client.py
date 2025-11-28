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
# SINAIS DEVEM ESTAR NA MESMA PASTA
SIGNAL60 = ['sinal_1_60x60.csv', 'sinal_2_60x60.csv', 'sinal_3_60x60.csv']
SIGNAL30 = ['sinal_1_30x30.csv', 'sinal_2_30x30.csv', 'sinal_3_30x30.csv']
ALGORITHM = ['CGNE', 'CGNR']

def make_request(target_url, server_tag, sinal_bin, tamanho, model, signal, algorithm, gain):
    try:
        # headers explicados
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

            print(f"[SUCESSO] {server_tag.upper()} - {image_name} salva em {req_duration:.2f}s")

        else:
            print(f"[ERRO] {server_tag.upper()} - Resposta: {resp.status_code} - {resp.text}")

    except requests.exceptions.RequestException as e:
        print(f"[ERRO] {server_tag.upper()} - Falha na comunicação: {e}")


def send_signal(index):
    model = random.choice(MODELS)
    algorithm = random.choice(ALGORITHM)    
    if model == 'H_60x60.csv':
        filename = random.choice(SIGNAL60)
        S = 794
    else:
        filename = random.choice(SIGNAL30)
        S = 436
    try:
        raw_signal = np.loadtxt(filename, delimiter=",").flatten()
    except Exception as e:
        print(f"[ERRO] Não foi possível ler {filename}: {e}")
        return []
    
    N_SENSORS = 64

    y_indices = np.arange(S, dtype=np.float64)
    
    gamma_sensor = 100.0 + 0.05 * y_indices * np.sqrt(y_indices)
    
    gamma_full = np.tile(gamma_sensor, N_SENSORS)
    
    signal_gain = raw_signal * gamma_full
    
    gain_str = "Formula_100_plus_005_y_sqrty"
    
    sinal_bin = signal_gain.astype(np.float32).tobytes()
    tamanho = len(signal_gain)

    print(f"[DISPARO {index}] Enviando {filename} (S={S}, N=64)...")

    threads_criadas = []

   
    thread_python = threading.Thread(
            target=make_request,
            args=(URL_PYTHON_SERVER, "python", sinal_bin, tamanho, model, filename, algorithm, gain_str)
        )

    thread_java = threading.Thread(
            target=make_request,
            args=(URL_JAVA_SERVER, "java", sinal_bin, tamanho, model, filename, algorithm, gain_str)
        )

    thread_python.start()
    thread_java.start()

    threads_criadas.append(thread_python)
    threads_criadas.append(thread_java)

    return threads_criadas

def executar_cliente(num_sinais=10):
    print(f"=== INICIANDO TESTE DE CARGA: {num_sinais} SINAIS ===")
    print(f"Total esperado de requisições: {num_sinais * 2 * 2}")

    with open('relatorio_imagens.txt', 'w') as f:
        f.write("=== RELATÓRIO ===\n")
        f.write(f"Inicio Teste: {datetime.datetime.now()}\n\n")

    with open('relatorio_desempenho.txt', 'w') as f:
        f.write("=== DESEMPENHO ===\n")
        f.write(f"Inicio Teste: {datetime.datetime.now()}\n\n")

    all_threads = []
    start_time = time.time()

    for i in range(num_sinais):
        novas_threads = send_signal(i+1)
        all_threads.extend(novas_threads)
        time.sleep(random.randint(0,5)) 

    print(f"\n=== TODOS OS SINAIS FORAM DISPARADOS ===")
    print(f"=== AGUARDANDO RETORNO... ===\n")

    for t in all_threads:
        t.join()

    total_time = time.time() - start_time
    print(f"=== TESTE DE CARGA FINALIZADO EM {total_time:.2f} SEGUNDOS ===")

if __name__ == "__main__":
    executar_cliente(num_sinais=10)