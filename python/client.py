import requests
import random
import time
import threading
import os
import datetime
import numpy as np
import uuid

URL_PYTHON_SERVER = "http://localhost:5000/interpretedServer/reconstruct"
URL_JAVA_SERVER = "http://localhost:8080/compiledServer/reconstruct"

file_lock = threading.Lock()

def read_sorteio_file(filename='sorteio_requisicoes.txt'): 
    requests_list = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    model, signal, algorithm, has_gain_str = parts
                    has_gain = has_gain_str.lower() == 'true' 
                    
                    S = 794 if model == 'H_60x60.csv' else 436
                    
                    requests_list.append({
                        'model': model,
                        'signal': signal,
                        'algorithm': algorithm,
                        'has_gain': has_gain,
                        'S': S
                    })
        print(f"[INFO] {len(requests_list)} requisições lidas do arquivo: {filename}")
    except FileNotFoundError:
        print(f"[ERRO] Arquivo de sorteio não encontrado: {filename}. Execute 'sorteio.py' primeiro.")
    except Exception as e:
        print(f"[ERRO] Erro ao ler o arquivo {filename}: {e}")
        
    return requests_list

def make_request(target_url, server_tag, sinal_bin, tamanho, model, signal, algorithm, gain,report_img_path, report_perf_path, output_dir):
    try:
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
            image_name = f"img_{algorithm}_{signal.replace('.csv', '')}_{server_tag}.png"
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
                with open(report_img_path, 'a') as f:
                    f.write(
                        f"{image_name} - Arquivo: {alg}, "
                        f"Inicio: {start}, Fim: {finish}, Tamanho: {size}, "
                        f"Iterações: {iteracoes}, Tempo Servidor: {exec_time} s, "
                        f"Tempo Req Total: {req_duration:.4f} s, Modelo: {model}, "
                        f"Sinal: {signal}, Ganho: {gain}\n"
                    )

            with file_lock:
                with open(report_perf_path, 'a') as f:
                    f.write(
                        f"[{finish}] Servidor: {server_tag.upper()}, CPU: {uso_cpu}%, Memória: {uso_mem}%\n"
                    )

            print(f"[SUCESSO] {server_tag.upper()} - {image_name} salva em {req_duration:.2f}s")

        else:
            print(f"[ERRO] {server_tag.upper()} - Resposta: {resp.status_code} - {resp.text}")

    except requests.exceptions.RequestException as e:
        print(f"[ERRO] {server_tag.upper()} - Falha na comunicação: {e}")

def get_server_choice():
    print("\n==============================================")
    print("Para qual servidor(es) você deseja enviar as requisições?")
    print("1 - Servidor Python")
    print("2 - Servidor Java")
    print("==============================================")
    
    while True:
        choice = input("Digite 1 ou 2: ").strip()
        if choice in ['1', '2']:
            return choice
        print("Opção inválida. Por favor, digite 1 ou 2")

def send_signal(index, params, report_img_path, report_perf_path, output_dir, server_choice):
    model = params['model']
    filename = params['signal']
    algorithm = params['algorithm']
    S = params['S']
    has_gain = params['has_gain']

    try:
        raw_signal = np.loadtxt(filename, delimiter=",").flatten()
    except Exception as e:
        print(f"[ERRO] Não foi possível ler {filename}: {e}")
        return []
    
    N_SENSORS = 64

    if has_gain:
        y_indices = np.arange(S, dtype=np.float64)
        gamma_sensor = 100.0 + 0.05 * y_indices * np.sqrt(y_indices)
        gamma_full = np.tile(gamma_sensor, N_SENSORS)
        signal_gain = raw_signal * gamma_full
        gain_str = "Formula_100_plus_005_y_sqrty"
    else:
        signal_gain = raw_signal 
        gain_str = "Nulo"

    sinal_bin = signal_gain.astype(np.float32).tobytes()
    tamanho = len(signal_gain)

    print(f"[DISPARO {index}] Enviando {filename} (S={S}, N=64)...")

    threads_criadas = []

    if server_choice in ['1']:
        thread_python = threading.Thread(
            target=make_request,
            args=(URL_PYTHON_SERVER, "python", sinal_bin, tamanho, model, filename, algorithm, gain_str, report_img_path, report_perf_path, output_dir)
        )
        thread_python.start()
        threads_criadas.append(thread_python)
        
    if server_choice in ['2']:
        thread_java = threading.Thread(
            target=make_request,
            args=(URL_JAVA_SERVER, "java", sinal_bin, tamanho, model, filename, algorithm, gain_str, report_img_path, report_perf_path, output_dir)
        )
        thread_java.start()
        threads_criadas.append(thread_java)

    if not threads_criadas:
         print(f"[AVISO] Nenhum thread iniciado. Escolha inválida ({server_choice}).")

    return threads_criadas

def executar_cliente(sorteio_filename='sorteio_requisicoes.txt'):
    
    server_choice = get_server_choice()

    requests_to_execute = read_sorteio_file(sorteio_filename)
    if not requests_to_execute:
        print("[ERRO] Nenhuma requisição para executar. Saindo.")
        return

    num_sinais = len(requests_to_execute)

    client_id = str(uuid.uuid4())[:8]
    report_img_path = f'relatorio_imagens_{client_id}.txt'
    report_perf_path = f'relatorio_desempenho_{client_id}.txt'
    output_dir = f"Reconstructed_{client_id}"

    print(f"=== INICIANDO CLIENTE ID: {client_id} ===")
    print(f"=== TESTE DE CARGA: {num_sinais} SINAIS ===")
    print(f"Relatórios serão salvos em: {report_img_path} e {report_perf_path}")
    print(f"Imagens serão salvas em: {output_dir}/")

    with open(report_img_path, 'w') as f:
        f.write("=== RELATÓRIO ===\n")
        f.write(f"Inicio Teste: {datetime.datetime.now()}\n\n")

    with open(report_perf_path, 'w') as f:
        f.write("=== DESEMPENHO ===\n")
        f.write(f"Inicio Teste: {datetime.datetime.now()}\n\n")

    all_threads = []
    start_time = time.time()

    for i, params in enumerate(requests_to_execute):
        novas_threads = send_signal(i+1, params, report_img_path, report_perf_path, output_dir, server_choice)
        all_threads.extend(novas_threads)

    print(f"\n=== TODOS OS SINAIS FORAM DISPARADOS ===")
    print(f"=== AGUARDANDO RETORNO... ===\n")

    for t in all_threads:
        t.join()

    total_time = time.time() - start_time
    print(f"=== TESTE DE CARGA {client_id} FINALIZADO EM {total_time:.2f} SEGUNDOS ===")

if __name__ == "__main__":
    executar_cliente()