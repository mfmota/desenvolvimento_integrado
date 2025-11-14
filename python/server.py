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

DATA_CACHE = {} # Cache agora vai guardar dicionários
MODEL_FILES = ['H_60x60.csv','H_30x30.csv']
SIGNAL_FILES = [
    'sinal_1_60x60.csv', 'sinal_2_60x60.csv', 'sinal_3_60x60.csv',
    'sinal_1_30x30.csv', 'sinal_2_30x30.csv', 'sinal_3_30x30.csv'
]

### ALTERADO ###
def pre_load_data():
    print("=== INICIANDO PRÉ-CARREGAMENTO DE DADOS E CÁLCULOS ===")
    
    # 1. Carrega Sinais (eles não mudam)
    for f in SIGNAL_FILES:
        try:
            DATA_CACHE[f] = np.loadtxt(f, delimiter=',', dtype=np.float64)
            print(f" - [CACHE SINAL] {f} carregado.")
        except Exception as e:
            print(f"[ERRO CACHE] Falha ao carregar {f}. Erro: {e}")

    # 2. Carrega Modelos e pré-calcula tudo
    for f in MODEL_FILES:
        try:
            print(f" - [CACHE MODELO] Processando {f}...")
            H = np.loadtxt(f, delimiter=',', dtype=np.float64)
            
            # Pré-calcula normalização de H
            H_mean = np.mean(H)
            H_std = np.std(H)
            if H_std > 1e-12:
                H_norm = (H - H_mean) / H_std
            else:
                H_norm = H - H_mean
            
            H_norm_T = H_norm.T # Pré-calcula transposta
            
            # Pré-calcula 'c' (O CÁLCULO LENTO)
            print(f"   -> Calculando 'c' para {f} (pode demorar)...")
            start_c = time.time()
            c_factor = np.linalg.norm(H_norm_T @ H_norm, ord=2)
            end_c = time.time()
            print(f"   -> 'c' calculado ({c_factor:.4e}) em {end_c - start_c:.2f}s")
            
            # Guarda tudo no cache
            DATA_CACHE[f] = {
                'H_norm': H_norm,
                'H_norm_T': H_norm_T,
                'H_mean': H_mean,
                'H_std': H_std,
                'c_factor': c_factor
            }
            print(f" - [CACHE MODELO] {f} processado e armazenado.")
            
        except Exception as e:
            print(f"[ERRO CACHE] Falha ao carregar {f}. Erro: {e}")
            
    # Inicializa o psutil
    psutil.cpu_percent(interval=None) 
    print("=== PRÉ-CARREGAMENTO CONCLUÍDO ===")

MAX_ITERATIONS = 10
ERROR_TOLERANCE = 1e-4

### ALTERADO ###
# Funções agora recebem H_norm e H_norm_T (transposta)
# para evitar recalcular a transposta no loop
def execute_cgne(H_norm: np.ndarray, H_norm_T: np.ndarray, g_norm: np.ndarray):
    
    m, n = H_norm.shape
    print(f"Resolvendo sistema {m}x{n} (CGNE) com {MAX_ITERATIONS} iterações máximas")

    f = np.zeros(n)
    r = g_norm.copy() # .copy() é necessário aqui pois 'r' é modificado
    
    p = H_norm_T @ r # Usa H_norm_T pré-calculada
    r_norm_sq_old = np.dot(r, r)

    i = 0 
    for i in range(MAX_ITERATIONS):
        p_norm_sq = np.dot(p, p)
        if p_norm_sq < 1e-20:
            break
            
        alpha = r_norm_sq_old / p_norm_sq
        f_next = f + alpha * p
        q = H_norm @ p 
        r_next = r - alpha * q
        
        error_absolute = np.linalg.norm(r_next, ord=2)
        error_relative = abs(error_absolute - np.linalg.norm(r, ord=2))

        if error_absolute < ERROR_TOLERANCE or error_relative < ERROR_TOLERANCE:
            f = f_next 
            break
            
        r_norm_sq_new = np.dot(r_next, r_next)
        if r_norm_sq_old < 1e-20:
            f = f_next
            break

        beta = r_norm_sq_new / r_norm_sq_old
        p_next = (H_norm_T @ r_next) + beta * p # Usa H_norm_T
        
        f = f_next
        r = r_next
        p = p_next
        r_norm_sq_old = r_norm_sq_new 
        
    print(f"Reconstrução CGNE concluída em {i+1} iterações")
    return f, i + 1

### ALTERADO ###
def execute_cgnr(H_norm: np.ndarray, H_norm_T: np.ndarray, g_norm: np.ndarray):

    m, n = H_norm.shape
    print(f"Resolvendo sistema {m}x{n} (CGNR) com {MAX_ITERATIONS} iterações máximas")

    f = np.zeros(n)
    r = g_norm - H_norm @ f
    z = H_norm_T @ r # Usa H_norm_T
    p = z.copy() # .copy() necessário

    for i in range(MAX_ITERATIONS):
        w = H_norm @ p
        w_norm_sq = np.linalg.norm(w, ord=2) ** 2
        z_norm_sq = np.linalg.norm(z, ord=2) ** 2
        
        if w_norm_sq < 1e-20:
             break
        
        alpha = z_norm_sq / w_norm_sq
        f_next = f + alpha * p
        r_next = r - alpha * w
        error_absolute = np.linalg.norm(r_next, ord=2)
        error_relative = abs(error_absolute - np.linalg.norm(r, ord=2))
        
        if error_absolute < ERROR_TOLERANCE or error_relative < ERROR_TOLERANCE:
            f = f_next
            break
            
        z_next = H_norm_T @ r_next # Usa H_norm_T
        z_next_norm_sq = np.linalg.norm(z_next, ord=2) ** 2
        
        if z_norm_sq < 1e-20:
            break
            
        beta = z_next_norm_sq / z_norm_sq
        p_next = z_next + beta * p
        
        f = f_next
        r = r_next
        z = z_next
        p = p_next

    print(f"Reconstrução CGNR concluída em {i+1} iterações")
    return f, i+1


@app.post("/interpretedServer/reconstruct")
def reconstruct():
    with semaforo_clientes:
        data = request.json
        algorithm = data['algoritmo']
        model_name = data['modelo']
        sinal_name = data['sinal']

        start_time = time.time()
        start_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with semaforo_processos:
            try:
                ### ALTERADO: Busca dados do cache (SEM .copy() para H e g) ###
                
                # Pega dados do modelo (pré-processados)
                model_data = DATA_CACHE.get(model_name)
                if model_data is None:
                    return jsonify({'error': f"Modelo {model_name} não encontrado no cache."}), 404
                
                H_norm = model_data['H_norm']
                H_norm_T = model_data['H_norm_T']
                H_std = model_data['H_std']
                c_factor = model_data['c_factor'] # Pega 'c' pré-calculado

                # Pega dados do sinal (bruto)
                g = DATA_CACHE.get(sinal_name)
                if g is None:
                    return jsonify({'error': f"Sinal {sinal_name} não encontrado no cache."}), 404
        
                # ===============================================
                ### ALTERADO: LÓGICA DE CÁLCULO MUITO MAIS SIMPLES ###
                # ===============================================

                # --- 1. Cálculo de Ganho (Apenas log, não aplicado) ---
                # (Lógica de S/N removida para simplificar, já que não é usada)
                
                # --- 2. Normalização de 'g' (A única feita por requisição) ---
                g_mean = np.mean(g)
                g_std = np.std(g)
                if g_std > 1e-12:
                    g_norm = (g - g_mean) / g_std
                else:
                    g_norm = g - g_mean
                
                # --- 3. Fator de Redução (c) ---
                print(f"[CÁLCULO] Fator de redução c (pré-calculado): {c_factor:.4e}")

                # --- 4. Coeficiente de Regularização (λ) ---
                # Este ainda precisa ser calculado por requisição, pois depende de 'g'
                lambda_reg = np.max(np.abs(H_norm_T @ g_norm)) * 0.10
                print(f"[CÁLCULO] Coeficiente λ: {lambda_reg:.4e}")
                
                # ===============================================
                ### FIM DAS ALTERAÇÕES ###
                # ===============================================

                # 5. Executa o algoritmo
                if "CGNE".lower() == algorithm.lower():
                    f,iteracoes = execute_cgne(H_norm, H_norm_T, g_norm)
                elif "CGNR".lower() == algorithm.lower():
                    f,iteracoes = execute_cgnr(H_norm, H_norm_T, g_norm)
                else:
                    return jsonify({'error': f"Algoritmo {algorithm} desconhecido"}), 400

                ### ALTERADO: psutil não-bloqueante ###
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=None) # NÃO BLOQUEIA MAIS

                # 6. De-normalização
                if H_std > 1e-12:
                    f_final = f * (g_std / H_std)
                else:
                    f_final = f
                
                # 7. Geração da Imagem
                f_clipped = np.clip(f_final, 0, None)
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