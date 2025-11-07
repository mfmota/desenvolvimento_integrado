import time
import numpy as np
import psutil
import io
from PIL import Image
import matplotlib.pyplot as plt
import threading
import datetime
from flask import Flask, request, jsonify, send_file, make_response

app = Flask(__name__)
semaforo_clientes = threading.Semaphore(2)
semaforo_processos = threading.Semaphore(5)

DATA_CACHE = {}
MODEL_FILES = ['H_60x60.csv','H_30x30.csv']
SIGNAL_FILES = [
    'sinal_1_60x60.csv', 'sinal_2_60x60.csv', 'sinal_3_60x60.csv',
    'sinal_1_30x30.csv', 'sinal_2_30x30.csv', 'sinal_3_30x30.csv'
]

def pre_load_data():
    print("=== INICIANDO PRÉ-CARREGAMENTO DE DADOS ===")
    all_files = list(set(MODEL_FILES + SIGNAL_FILES)) 
    for f in all_files:
        try:
            DATA_CACHE[f] = np.loadtxt(f, delimiter=',', dtype=np.float64)
            print(f" - [CACHE] {f} carregado com sucesso.")
        except Exception as e:
            print(f"[ERRO CACHE] Falha ao carregar {f}. Servidor pode falhar. Erro: {e}")
    print("=== PRÉ-CARREGAMENTO CONCLUÍDO ===")

MAX_ITERATIONS = 10
ERROR_TOLERANCE = 1e-4

def execute_cgne(H: np.ndarray, g: np.ndarray):
    g_mean = np.mean(g)
    g_std = np.std(g)
    if g_std > 1e-12:
        g_norm = (g - g_mean) / g_std
    else:
        g_norm = g - g_mean
        print("Aviso: Desvio padrão de g muito pequeno, normalização parcial aplicada.")

    H_mean = np.mean(H)
    H_std = np.std(H)
    if H_std > 1e-12:
        H_norm = (H - H_mean) / H_std
    else:
        H_norm = H - H_mean
        print("Aviso: Desvio padrão de H muito pequeno, normalização parcial aplicada.")

    m, n = H_norm.shape
    print(f"Resolvendo sistema {m}x{n} com {MAX_ITERATIONS} iterações máximas (normalização z-score) - CGNE")

    f = np.zeros(n)
    r = g_norm.copy()
    
    p = H_norm.T @ r
    r_norm_sq_old = np.dot(r, r)

    i = 0 
    for i in range(MAX_ITERATIONS):
        
        p_norm_sq = np.dot(p, p)
        
        if p_norm_sq < 1e-20:
            print(f"Convergência (p_norm_sq ~ 0) na iteração {i+1}")
            break
            
        alpha = r_norm_sq_old / p_norm_sq
        f_next = f + alpha * p
        
        q = H_norm @ p 
        r_next = r - alpha * q
        
        error_absolute = np.linalg.norm(r_next, ord=2)
        error_relative = abs(error_absolute - np.linalg.norm(r, ord=2)) # r ainda é r_i

        if error_absolute < ERROR_TOLERANCE or error_relative < ERROR_TOLERANCE:
            print(f"Convergência atingida na iteração {i+1}")
            print(f"Erro absoluto: {error_absolute:.2e}, Erro relativo: {error_relative:.2e}")
            f = f_next 
            break
            
        r_norm_sq_new = np.dot(r_next, r_next)
        
        if r_norm_sq_old < 1e-20:
            print(f"Convergência (r_norm_sq_old ~ 0) na iteração {i+1}")
            f = f_next
            break

        beta = r_norm_sq_new / r_norm_sq_old
        p_next = (H_norm.T @ r_next) + beta * p
        
        f = f_next
        r = r_next
        p = p_next
        r_norm_sq_old = r_norm_sq_new 
        if (i + 1) % 10 == 0:
            print(f"Iteração {i+1}: erro = {error_absolute:.2e}")

    if H_std > 1e-12:
        f_final = f * (g_std / H_std)
    else:
        f_final = f
        
    print(f"Reconstrução CGNE concluída em {i+1} iterações (z-score)")
    return f_final, i + 1

def execute_cgnr(H: np.ndarray, g: np.ndarray):
    g_mean = np.mean(g)
    g_std = np.std(g)
    if g_std > 1e-12:
        g_norm = (g - g_mean) / g_std
    else:
        g_norm = g - g_mean
        print("Aviso: Desvio padrão de g muito pequeno, normalização parcial aplicada.")

    H_mean = np.mean(H)
    H_std = np.std(H)

    if H_std > 1e-12:
        H_norm = (H - H_mean) / H_std
    else:
        H_norm = H - H_mean
        print("Aviso: Desvio padrão de H muito pequeno, normalização parcial aplicada.")

    m, n = H_norm.shape
    print(f"Resolvendo sistema {m}x{n} com {MAX_ITERATIONS} iterações máximas (normalização z-score)")

    f = np.zeros(n)
    r = g_norm - H_norm @ f

    z = H_norm.T @ r
    
    p = z.copy()

    for i in range(MAX_ITERATIONS):
        w = H_norm @ p
        
        w_norm_sq = np.linalg.norm(w, ord=2) ** 2
            
        z_norm_sq = np.linalg.norm(z, ord=2) ** 2
        
        alpha = z_norm_sq / w_norm_sq
        
        f_next = f + alpha * p
        
        r_next = r - alpha * w
        error_absolute = np.linalg.norm(r_next, ord=2)
        error_relative = abs(error_absolute - np.linalg.norm(r, ord=2))
        
        if error_absolute < ERROR_TOLERANCE or error_relative < ERROR_TOLERANCE:
            print(f"Convergência atingida na iteração {i+1}")
            print(f"Erro absoluto: {error_absolute:.2e}, Erro relativo: {error_relative:.2e}")
            f = f_next
            break
            
        z_next = H_norm.T @ r_next
        z_next_norm_sq = np.linalg.norm(z_next, ord=2) ** 2
            
        beta = z_next_norm_sq / z_norm_sq
        p_next = z_next + beta * p
        
        f = f_next
        r = r_next
        z = z_next
        p = p_next

        if (i + 1) % 10 == 0:
            print(f"Iteração {i+1}: erro = {error_absolute:.2e}")
    if H_std > 1e-12:
        f_final = f * (g_std / H_std)
    else:
        f_final = f
        
    print(f"Reconstrução concluída em {i+1} iterações (z-score)")
    return f_final, i+1


@app.post("/interpretedServer/reconstruct")
def reconstruct():
    with semaforo_clientes:
        data = request.json
        algorithm = data['algoritmo']
        model = data['modelo']
        sinal = data['sinal']

        start_time = time.time()
        start_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with semaforo_processos:
            try:
                H = DATA_CACHE.get(model)
                g = DATA_CACHE.get(sinal)

                if H is None or g is None:
                    print(f"[ERRO] {model} ou {sinal} não encontrado no cache.")
                    return jsonify({'error': 'Dados não encontrados no cache do servidor'}), 404
        
                if "CGNE".lower() == algorithm.lower():
                    f,iteracoes = execute_cgne(H, g)
                elif "CGNR".lower() == algorithm.lower():
                    f,iteracoes = execute_cgnr(H, g)
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)

                f_clipped = np.clip(f, 0, None)
                
                f_max = f_clipped.max()

                if f_max > 1e-12: 
                   f_norm = (f_clipped / f_max) * 255
                else:
                   f_norm = np.full_like(f, 0)

                lado = int(np.sqrt(len(f)))
                imagem_array = f_norm[:lado*lado].reshape((lado, lado), order='F')

                imagem_array = np.clip(imagem_array, 0, 255)
                imagem = Image.fromarray(imagem_array.astype('uint8'))

                img_bytes = io.BytesIO()
                imagem.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                end_time = time.time()
                end_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                response = make_response(send_file(img_bytes, mimetype='image/png', download_name='reconstruida.png'))
                response.headers['X-Algoritmo'] = algorithm 
                response.headers['X-Inicio'] = start_dt
                response.headers['X-Fim'] = end_dt
                response.headers['X-Tamanho'] = f"{lado}x{lado}"
                response.headers['X-Iteracoes'] = str(iteracoes)
                response.headers['X-Tempo'] = str(end_time - start_time) 
                response.headers['X-Cpu'] = str(cpu)
                response.headers['X-Mem'] = str(mem.percent)

                return response
            except Exception as e:
                return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=["GET"])
def ping():
    return 'OK', 200

pre_load_data()
if __name__ == '__main__':
    print("Iniciando servidor Flask (desenvolvimento)")
    app.run(host='0.0.0.0', port=5000, threaded=True)