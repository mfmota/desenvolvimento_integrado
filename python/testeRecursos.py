import time
import numpy as np
import psutil
import os
import datetime
import threading

MAX_ITERATIONS = 10
ERROR_TOLERANCE = 1e-4
CACHE_DIR = "model_cache"

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

def load_h_matrices(model_name):
    base_name = os.path.splitext(model_name)[0]
    npy_path = os.path.join(CACHE_DIR, f"{base_name}.npy")
    npy_T_path = os.path.join(CACHE_DIR, f"{base_name}_T.npy")

    if not os.path.exists(npy_path) or not os.path.exists(npy_T_path):
        raise FileNotFoundError(f"Arquivos binários do modelo {model_name} não encontrados. Execute o servidor Flask uma vez para criá-los.")
    
    H_norm = np.load(npy_path)
    H_norm_T = np.load(npy_T_path)
    return H_norm, H_norm_T

def load_signal(filename):
    try:
        raw_signal = np.loadtxt(filename, delimiter=",").flatten()
    except FileNotFoundError:
        print(f"[ERRO] Sinal de teste {filename} não encontrado. Certifique-se de que seus arquivos de sinal .csv estão presentes.")
        return None
    
    g = raw_signal.astype(np.float64)
    g_mean = np.mean(g)
    g_std = np.std(g)
    
    if g_std > 1e-12:
        g_norm = (g - g_mean) / g_std
    else:
        g_norm = g - g_mean
    
    return g_norm, len(g)

def monitorar_recurso(target_func, args):
    proc = psutil.Process(os.getpid())
    
    mem_antes = proc.memory_info().rss / (1024 * 1024) # RAM em MB
    
    cpu_peak = 0.0
    mem_peak = mem_antes
    
    def worker():
        try:
            target_func(*args)
        except Exception as e:
            print(f"[ERRO THREAD] {e}")

    test_thread = threading.Thread(target=worker)
    test_thread.start()

    while test_thread.is_alive():
        try:
            cpu_current = proc.cpu_percent(interval=0.1)
            mem_current = proc.memory_info().rss / (1024 * 1024)
            cpu_peak = max(cpu_peak, cpu_current)
            mem_peak = max(mem_peak, mem_current)
        except psutil.NoSuchProcess:
            break
        
    test_thread.join()

    ram_diff = mem_peak - mem_antes
    
    return cpu_peak, ram_diff, mem_peak

def executar_teste_de_recursos():
    TEST_MAP = [
        ('H_60x60.csv', 'sinal_1_60x60.csv', 'cgne'), 
        ('H_60x60.csv', 'sinal_1_60x60.csv', 'cgnr'),

        ('H_60x60.csv', 'sinal_2_60x60.csv', 'cgne'), 
        ('H_60x60.csv', 'sinal_2_60x60.csv', 'cgnr'),

        ('H_60x60.csv', 'sinal_3_60x60.csv', 'cgne'), 
        ('H_60x60.csv', 'sinal_3_60x60.csv', 'cgnr'),

        ('H_30x30.csv', 'sinal_1_30x30.csv', 'cgne'), 
        ('H_30x30.csv', 'sinal_1_30x30.csv', 'cgnr'),

        ('H_30x30.csv', 'sinal_2_30x30.csv', 'cgne'), 
        ('H_30x30.csv', 'sinal_2_30x30.csv', 'cgnr'),

        ('H_30x30.csv', 'sinal_3_30x30.csv', 'cgne'), 
        ('H_30x30.csv', 'sinal_3_30x30.csv', 'cgnr'),

    ]

    print("=========================================================")
    print(f"       INICIANDO TESTE DE CONSUMO DE RECURSOS        ")
    print(f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cores de CPU: {psutil.cpu_count(logical=True)}")
    print(f"RAM Total (GB): {psutil.virtual_memory().total / (1024**3):.2f}")
    print("---------------------------------------------------------")
    print("{:<15} {:<15} {:<8} {:<10} {:<10} {:<10}".format(
        "MODELO", "SINAL", "ALGORITMO", "CPU MAX (%)", "RAM DIFF (MB)", "TEMPO (s)"))
    print("---------------------------------------------------------")

    resultados = []

    for model_name, signal_file, algorithm in TEST_MAP:
        try:
            H_norm, H_norm_T = load_h_matrices(model_name)
            signal_data = load_signal(signal_file)
            if signal_data is None:
                continue

            g_norm, tamanho = signal_data
            
            # Escolhe a função de execução
            exec_func = execute_cgne if algorithm.lower() == 'cgne' else execute_cgnr

            start_exec = time.time()
            # Monitora o recurso durante a execução
            cpu_peak, ram_diff, ram_peak_total = monitorar_recurso(exec_func, (H_norm, H_norm_T, g_norm))
            end_exec = time.time()
            
            tempo_exec = end_exec - start_exec
            
            resultados.append((model_name, signal_file, algorithm, cpu_peak, ram_diff, tempo_exec))
            
            print("{:<15} {:<15} {:<8} {:<10.1f} {:<10.1f} {:<10.2f}".format(
                model_name, signal_file, algorithm.upper(), cpu_peak, ram_diff, tempo_exec))
            
            del H_norm, H_norm_T # Liberação explícita de memória após cada teste

        except Exception as e:
            print(f"[ERRO GERAL] Falha no teste {model_name}/{signal_file}/{algorithm}: {e}")
            
        time.sleep(1) # Pequena pausa para estabilizar a CPU entre os testes

    print("=========================================================")
    print("\nRECOMENDAÇÕES DE DIMENSIONAMENTO:")
    print("1. CPU Max: O pico de CPU indica o uso máximo de um núcleo (ou mais se o NumPy paralelizar).")
    print("2. RAM Diff (MB): Representa a memória **adicional** que o processo alocou para este cálculo (além do JIT load).")
    
    if resultados:
        max_ram_diff = max([r[4] for r in resultados])
        print(f"\nRAM MÁXIMA ADICIONAL por processo: {max_ram_diff:.1f} MB")
        
        # Fórmula simplificada: RAM_Livre / RAM_Diff_Max
        ram_livre = (psutil.virtual_memory().available / (1024 * 1024))
        max_processos_estimado = int(ram_livre / max_ram_diff) if max_ram_diff > 0 else psutil.cpu_count()
        
        print(f"\nRAM Livre no momento: {ram_livre:.1f} MB")
        print(f"Número Máximo Teórico de Processos (por RAM): {max_processos_estimado}")
        print(f"Número de Cores de CPU: {psutil.cpu_count(logical=False)}")
        
        # Sugestão conservadora: 1 ou 2 processos por core, limitado pela RAM.
        sugestao_cpu = psutil.cpu_count(logical=False) * 1.5 
        
        limite_semaforo = min(sugestao_cpu, max_processos_estimado)
        
        print(f"\nSUGESTÃO PARA semaforo_processos: Mínimo entre o limite de CPU ({sugestao_cpu:.0f}) e o limite de RAM ({max_processos_estimado}).")
        print(f"Ajuste conservador recomendado para o Semáforo de Processamento: {max(2, int(limite_semaforo))}")
    
    print("=========================================================")


if __name__ == '__main__':
    executar_teste_de_recursos()