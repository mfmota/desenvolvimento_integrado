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
modelos = {}

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
                if model in modelos:
                    H = modelos[model]
                else:
                    H = np.loadtxt(model, delimiter=',', dtype=np.float64)
                    modelos[model] = H

                g = np.loadtxt(sinal, delimiter=',', dtype=np.float64)

                if "CGNE".lower() == algorithm.lower():
                    f,iteracoes = execute_cgne(H, g)
                elif "CGNR".lower() == algorithm.lower():
                    f,iteracoes = execute_cgnr(H, g)
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)

                f_min, f_max = f.min(), f.max()
                if f_max != f_min:
                    f_norm = (f - f_min) / (f_max - f_min) * 255
                else:
                    f_norm = np.full_like(f, 128)
        
                lado = int(np.sqrt(len(f)))
                imagem_array = f_norm[:lado*lado].reshape((lado, lado), order='F')

                imagem_array = np.clip(imagem_array, 0, 255)
                imagem = Image.fromarray(imagem_array.astype('uint8'))

                # Converte imagem para bytes
                img_bytes = io.BytesIO()
                imagem.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                # REQUISITO ATENDIDO: Data e hora do término da reconstrução
                end_time = time.time()
                end_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # REQUISITO ATENDIDO: Resposta com todos os metadados obrigatórios
                response = make_response(send_file(img_bytes, mimetype='image/png', download_name='reconstruida.png'))
                response.headers['X-Algoritmo'] = "cgnr"  # 2. Identificação do algoritmo
                response.headers['X-Inicio'] = start_dt  # 3. Data/hora início
                response.headers['X-Fim'] = end_dt  # 4. Data/hora término  
                response.headers['X-Tamanho'] = f"{lado}x{lado}"  # 5. Tamanho em pixels
                response.headers['X-Iteracoes'] = str(iteracoes)  # 6. Número de iterações
                response.headers['X-Tempo'] = str(end_time - start_time) 
                response.headers['X-Cpu'] = str(cpu)
                response.headers['X-Mem'] = str(mem.percent)

                return response
            except Exception as e:
                return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=["GET"])
def ping():
    return 'OK', 200

if __name__ == '__main__':
    print("Iniciando servidor")
    app.run(host='0.0.0.0', port=5000, threaded=True)