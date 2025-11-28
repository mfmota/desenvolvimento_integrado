import time
import numpy as np
import psutil
import io
from PIL import Image
import threading
import datetime
from flask import Flask, request, jsonify, send_file, make_response

app = Flask(__name__)
semaforo_clientes = threading.Semaphore(20)
semaforo_processos = threading.Semaphore(4)

DATA_CACHE = {}
MODEL_FILES = ['H_60x60.csv', 'H_30x30.csv']

def pre_load_models():
    print("=== INICIANDO PRÉ-CARREGAMENTO DE MODELOS ===")
    for f in MODEL_FILES:
        try:
            H = np.loadtxt(f, delimiter=',', dtype=np.float64)
            H_mean = np.mean(H)
            H_std = np.std(H)
            if H_std > 1e-12:
                H_norm = (H - H_mean) / H_std
            else:
                H_norm = H - H_mean
            H_norm_T = H_norm.T
            print(f"   -> Calculando 'c' para {f} (pode demorar)...")
            start_c = time.time()
            c_factor = np.linalg.norm(H_norm_T @ H_norm, ord=2)
            end_c = time.time()
            print(f"   -> 'c' calculado ({c_factor:.4e}) em {end_c - start_c:.2f}s")
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
    psutil.cpu_percent(interval=None)
    print("=== PRÉ-CARREGAMENTO DE MODELOS CONCLUÍDO ===")

MAX_ITERATIONS = 10
ERROR_TOLERANCE = 1e-4  

def execute_cgne(H_norm: np.ndarray, H_norm_T: np.ndarray, g_norm: np.ndarray):
    m, n = H_norm.shape
    f = np.zeros(n, dtype=np.float64)
    r = g_norm.copy()
    p = H_norm_T @ r
    r_norm_old = np.linalg.norm(r, ord=2)

    for i in range(MAX_ITERATIONS):
        p_norm_sq = np.dot(p, p)
        if p_norm_sq < 1e-20:
            break
        alpha = (r_norm_old * r_norm_old) / p_norm_sq 
        f_next = f + alpha * p
        q = H_norm @ p
        r_next = r - alpha * q

        r_norm_new = np.linalg.norm(r_next, ord=2)
        epsilon = abs(r_norm_new - r_norm_old)

        if epsilon < ERROR_TOLERANCE or r_norm_new < ERROR_TOLERANCE:
            f = f_next
            return f, i + 1

        r_norm_sq_new = r_norm_new * r_norm_new
        if (r_norm_old * r_norm_old) < 1e-20:
            f = f_next
            return f, i + 1

        beta = r_norm_sq_new / (r_norm_old * r_norm_old)
        p_next = (H_norm_T @ r_next) + beta * p

        f = f_next
        r = r_next
        p = p_next
        r_norm_old = r_norm_new

    return f, MAX_ITERATIONS

def execute_cgnr(H_norm: np.ndarray, H_norm_T: np.ndarray, g_norm: np.ndarray):
    m, n = H_norm.shape
    f = np.zeros(n, dtype=np.float64)
    r = g_norm - H_norm @ f
    z = H_norm_T @ r
    p = z.copy()

    r_norm_old = np.linalg.norm(r, ord=2)
    z_norm_sq_old = np.linalg.norm(z, ord=2) ** 2

    for i in range(MAX_ITERATIONS):
        w = H_norm @ p
        w_norm_sq = np.linalg.norm(w, ord=2) ** 2
        if w_norm_sq < 1e-20:
            break

        alpha = z_norm_sq_old / w_norm_sq
        f_next = f + alpha * p
        r_next = r - alpha * w

        r_norm_new = np.linalg.norm(r_next, ord=2)
        epsilon = abs(r_norm_new - r_norm_old)
        if epsilon < ERROR_TOLERANCE or r_norm_new < ERROR_TOLERANCE:
            f = f_next
            return f, i + 1

        z_next = H_norm_T @ r_next
        z_norm_sq_new = np.linalg.norm(z_next, ord=2) ** 2

        if z_norm_sq_old < 1e-20:
            f = f_next
            return f, i + 1

        beta = z_norm_sq_new / z_norm_sq_old
        p_next = z_next + beta * p

        f = f_next
        r = r_next
        z = z_next
        p = p_next
        r_norm_old = r_norm_new
        z_norm_sq_old = z_norm_sq_new

    return f, MAX_ITERATIONS

@app.post("/interpretedServer/reconstruct")
def reconstruct():
    with semaforo_clientes:
        content_type = request.headers.get('Content-Type', '')
        if not content_type.startswith('application/octet-stream'):
            return jsonify({'error': 'Este servidor aceita apenas application/octet-stream (binário puro).'}), 400

        model_name = request.headers.get('X-Modelo')
        algorithm = request.headers.get('X-Alg') or request.headers.get('X-Algoritmo')
        tamanho_header = request.headers.get('X-Tamanho')
        ganho_header = request.headers.get('X-Ganho')

        if not model_name:
            return jsonify({'error': 'Header X-Modelo ausente.'}), 400
        if not algorithm:
            return jsonify({'error': 'Header X-Alg ausente.'}), 400

        start_time = time.time()
        start_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with semaforo_processos:
            try:
                model_data = DATA_CACHE.get(model_name)
                if model_data is None:
                    return jsonify({'error': f"Modelo {model_name} não encontrado no cache."}), 404

                H_norm = model_data['H_norm']
                H_norm_T = model_data['H_norm_T']
                H_std = model_data['H_std']
                c_factor = model_data['c_factor']

                raw_bytes = request.get_data()
                if not raw_bytes:
                    return jsonify({'error': 'Corpo binário vazio.'}), 400

                try:
                    sinal_float32 = np.frombuffer(raw_bytes, dtype=np.float32)
                except Exception as e:
                    return jsonify({'error': f'Falha ao decodificar bytes para float32: {e}'}), 400

                if tamanho_header:
                    try:
                        expected_len = int(tamanho_header)
                        if expected_len != len(sinal_float32):
                            return jsonify({'error': f"Tamanho incorreto: esperado {expected_len}, recebi {len(sinal_float32)}"}), 400
                    except ValueError:
                        pass

                g = sinal_float32.astype(np.float64)

                g_mean = np.mean(g)
                g_std = np.std(g)
                if g_std > 1e-12:
                    g_norm = (g - g_mean) / g_std
                else:
                    g_norm = g - g_mean

                print(f"[CÁLCULO] Fator de redução c (pré-calculado): {c_factor:.4e}")

                lambda_reg = np.max(np.abs(H_norm_T @ g_norm)) * 0.10
                print(f"[CÁLCULO] Coeficiente λ: {lambda_reg:.4e}")

                alg_lower = algorithm.strip().lower()
                if alg_lower == 'cgne':
                    f, iterations = execute_cgne(H_norm, H_norm_T, g_norm)
                elif alg_lower == 'cgnr':
                    f, iterations = execute_cgnr(H_norm, H_norm_T, g_norm)
                else:
                    return jsonify({'error': f"Algoritmo {algorithm} desconhecido"}), 400

                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=None)

                if H_std > 1e-12:
                    f_final = f * (g_std / H_std)
                else:
                    f_final = f

                f_clipped = np.clip(f_final, 0, None)
                f_max = f_clipped.max()
                if f_max > 1e-12:
                    f_norm = (f_clipped / f_max) * 255.0
                else:
                    f_norm = np.zeros_like(f_clipped)

                n_pixels = len(f_norm)
                lado = int(np.floor(np.sqrt(n_pixels)))
                if lado * lado == 0:
                    return jsonify({'error': 'Tamanho do vetor de reconstrução inválido.'}), 500

                img_array = f_norm[:lado*lado].reshape((lado, lado), order='F')
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                imagem = Image.fromarray(img_array)

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
                response.headers['X-Iteracoes'] = str(iterations)
                response.headers['X-Tempo'] = str(end_time - start_time)
                response.headers['X-Cpu'] = str(cpu)
                response.headers['X-Mem'] = str(mem.percent)
                if ganho_header:
                    response.headers['X-Ganho'] = ganho_header

                return response

            except Exception as e:
                return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=["GET"])
def ping():
    return 'OK', 200

if __name__ == '__main__':
    pre_load_models()
    print("Iniciando servidor Flask (desenvolvimento) - ouvindo na porta 5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)