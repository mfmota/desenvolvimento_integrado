import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import psutil
import io
import threading
import datetime
from flask import Flask, request, jsonify, send_file, make_response
from PIL import Image
from typing import Tuple, List

app = Flask(__name__)

test_rounds = 5
cpu_cap = 90.0
active_clients = 0
waiting_clients = 0

cgnr_cpus: List[float] = []
cgnr_mems: List[float] = []
cgne_cpus: List[float] = [] 
cgne_mems: List[float] = []

FILE_ID_MAP = {
    'H_60x60.csv': 0, 'H_30x30.csv': 1, 
    'sinal_1_30x30.csv': 0, 'sinal_2_30x30.csv': 1, 'sinal_3_30x30.csv': 2, 
    'sinal_1_60x60.csv': 3, 'sinal_2_60x60.csv': 4, 'sinal_3_60x60.csv': 5,
}

NUM_FILES_TESTED = 7 
semaphore_files = [threading.Semaphore(1) for _ in range(NUM_FILES_TESTED)]
semaphore7 = threading.Semaphore(1) 

RAW_MODEL_CACHE = {} 
MODEL_FILES = ['H_60x60.csv', 'H_30x30.csv']
CACHE_DIR = "model_cache"
MAX_ITERATIONS = 10
ERROR_TOLERANCE = 1e-4

def load_raw_models_ram():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    print("=== CARREGANDO MODELOS BRUTOS PARA RAM ===")
    
    for csv_file in MODEL_FILES:
        base_name = os.path.splitext(csv_file)[0]
        npy_path = os.path.join(CACHE_DIR, f"{base_name}.npy")
        
        try:
            if os.path.exists(npy_path):
                print(f" -> Lendo binário: {npy_path}")
                H = np.load(npy_path)
            else:
                print(f" -> Convertendo CSV para Binário (1ª vez): {csv_file}")
                H = np.loadtxt(csv_file, delimiter=',', dtype=np.float64)
                np.save(npy_path, H)

            RAW_MODEL_CACHE[csv_file] = H.astype(np.float32)
            
            print(f" -> {csv_file} carregado na RAM (Bruto).")
            
        except Exception as e:
            print(f"[ERRO] Falha ao carregar {csv_file}: {e}")

    print("=== CARREGAMENTO CONCLUÍDO ===")

def execute_cgne(H, g_norm):
    H_T = H.T 
    
    f = np.zeros(H.shape[1], dtype=np.float32)
    r = g_norm.astype(np.float32)
    p = H_T @ r
    r_norm_old = np.linalg.norm(r)

    for i in range(MAX_ITERATIONS):
        p_norm_sq = np.dot(p, p)
        if p_norm_sq < 1e-15: break
        
        alpha = (r_norm_old**2) / p_norm_sq 
        f = f + alpha * p
        r = r - alpha * (H @ p)
        r_norm_new = np.linalg.norm(r)

        if r_norm_new < ERROR_TOLERANCE: break
        
        beta = (r_norm_new**2) / (r_norm_old**2)
        p = (H_T @ r) + beta * p
        r_norm_old = r_norm_new
        
    return f, i + 1

def execute_cgnr(H, g_norm):
    H_T = H.T
    
    f = np.zeros(H.shape[1], dtype=np.float32)
    r = g_norm.astype(np.float32) - H @ np.zeros(H.shape[1], dtype=np.float32)
    z = H_T @ r
    p = z.copy()
    z_norm_sq_old = np.linalg.norm(z)**2

    for i in range(MAX_ITERATIONS):
        w = H @ p
        w_norm_sq = np.linalg.norm(w)**2
        if w_norm_sq < 1e-15: break

        alpha = z_norm_sq_old / w_norm_sq
        f = f + alpha * p
        r = r - alpha * w
        
        z_next = H_T @ r
        z_norm_sq_new = np.linalg.norm(z_next)**2
        
        if z_norm_sq_new < 1e-15: break

        beta = z_norm_sq_new / z_norm_sq_old
        p = z_next + beta * p
        z_norm_sq_old = z_norm_sq_new

    return f, i + 1

def _execute_alg_for_measurement(H_raw, g_raw, alg_name):
    start_measure = time.time()
    
    H_mean = np.mean(H_raw)
    H_std = np.std(H_raw)
    H_norm = (H_raw - H_mean) / H_std if H_std > 1e-12 else H_raw - H_mean
    
    g_mean = np.mean(g_raw)
    g_std = np.std(g_raw)
    g_norm = (g_raw - g_mean) / g_std if g_std > 1e-12 else g_raw - g_mean
    
    if alg_name == 'cgne':
        execute_cgne(H_norm, g_norm) 
    else:
        execute_cgnr(H_norm, g_norm)

    cpu_measure = psutil.cpu_percent(interval=None)
    mem_measure = psutil.virtual_memory().percent
    
    return cpu_measure, mem_measure

def determine_cpu_mem():
    global cgnr_cpus, cgnr_mems, cgne_cpus, cgne_mems
    
    if not RAW_MODEL_CACHE: return

    print("\n=== CALIBRANDO COM DADOS BRUTOS (Inclui tempo de cálculo matemático) ===")
    
    H_raw = RAW_MODEL_CACHE.get('H_60x60.csv', RAW_MODEL_CACHE.get('H_30x30.csv'))

    g_dummy = np.random.rand(H_raw.shape[0]).astype(np.float32)

    c_cpu, c_mem = _execute_alg_for_measurement(H_raw, g_dummy, 'cgnr')
    cgnr_cpus = [c_cpu] * NUM_FILES_TESTED
    cgnr_mems = [c_mem] * NUM_FILES_TESTED
    print(f" -> Estimativa CGNR (Total): CPU~{c_cpu:.1f}%")

    c_cpu, c_mem = _execute_alg_for_measurement(H_raw, g_dummy, 'cgne')
    cgne_cpus = [c_cpu] * NUM_FILES_TESTED
    cgne_mems = [c_mem] * NUM_FILES_TESTED
    print(f" -> Estimativa CGNE (Total): CPU~{c_cpu:.1f}%")

def client_wait(file_name, alg):
    global active_clients, waiting_clients
    permit = False
    
    resource_id = FILE_ID_MAP.get(file_name, 0)
    semaphore = semaphore_files[resource_id]
    semaphore.acquire()
    
    while not permit:
        semaphore7.acquire()
        current_cpu = psutil.cpu_percent(interval=None)
        
        if alg.lower() == 'cgnr':
            est = cgnr_cpus[0] if cgnr_cpus else 20.0
        else:
            est = cgne_cpus[0] if cgne_cpus else 20.0
            
        if active_clients == 0 or (current_cpu + est < cpu_cap * 1.3):
            permit = True
        
        semaphore7.release()
        if not permit: time.sleep(1)
            
    waiting_clients -= 1
    return resource_id

@app.post("/interpretedServer/reconstruct")
def reconstruct():
    global active_clients, waiting_clients
    
    model_name = request.headers.get('X-Modelo')
    algorithm = request.headers.get('X-Alg') or request.headers.get('X-Algoritmo')
    ganho_header = request.headers.get('X-Ganho')

    if not model_name or not algorithm: return jsonify({'error': 'Headers erro'}), 400

    waiting_clients += 1
    resource_id = client_wait(model_name, algorithm)
    active_clients += 1
    
    start_time = time.time()
    start_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        H_raw = RAW_MODEL_CACHE.get(model_name)
        if H_raw is None: return jsonify({'error': 'Modelo off'}), 404

        raw_bytes = request.get_data()
        g_raw = np.frombuffer(raw_bytes, dtype=np.float32)

        H_mean = np.mean(H_raw)
        H_std = np.std(H_raw)
        
        if H_std > 1e-12:
            H_norm = (H_raw - H_mean) / H_std
        else:
            H_norm = H_raw - H_mean
            
        g_mean = np.mean(g_raw)
        g_std = np.std(g_raw)
        g_norm = (g_raw - g_mean) / g_std if g_std > 1e-12 else g_raw - g_mean

        if algorithm.lower() == 'cgne':
            f, its = execute_cgne(H_norm, g_norm)
        else:
            f, its = execute_cgnr(H_norm, g_norm)

        if H_std > 1e-12:
            f = f * (g_std / H_std)
            
        f_clipped = np.clip(f, 0, None)
        f_max = f_clipped.max()
        f_norm_img = (f_clipped / f_max * 255.0) if f_max > 1e-9 else f_clipped
        
        lado = int(np.sqrt(len(f_norm_img)))
        img_arr = f_norm_img.reshape((lado, lado), order='F').astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format='PNG')
        buf.seek(0)
        
        end_time = time.time()
        
        resp = make_response(send_file(buf, mimetype='image/png'))
        resp.headers['X-Tempo'] = f"{end_time - start_time:.4f}"
        resp.headers['X-Algoritmo'] = algorithm
        resp.headers['X-Iteracoes'] = str(its)
        resp.headers['X-Cpu'] = str(psutil.cpu_percent(interval=None))
        resp.headers['X-Mem'] = str(psutil.virtual_memory().percent)
        if ganho_header: resp.headers['X-Ganho'] = ganho_header
        
        return resp

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        semaphore_files[resource_id].release()
        active_clients -= 1

if __name__ == '__main__':
    load_raw_models_ram()
    determine_cpu_mem()
    print("Servidor Pronto. (Cálculos em tempo real)")
    app.run(host='0.0.0.0', port=5000, threaded=True)