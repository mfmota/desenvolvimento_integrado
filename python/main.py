import client
import server
import time
import threading
from waitress import serve
import requests

def iniciar_servidor_python():
    """ Inicia o servidor Python (Flask/Waitress) """
    print("Iniciando servidor Python na porta 5000...")
    serve(server.app, host='0.0.0.0', port=5000, threads=5)

def aguardar_servidor(url, nome):
    """ Tenta se conectar a um servidor até que ele responda 200 OK. """
    print(f"Aguardando servidor {nome} em {url}...")
    while True:
        try:
            # Usamos um endpoint /ping que ambos os servidores devem ter
            r = requests.get(f"{url}/ping", timeout=1.0)
            if r.status_code == 200:
                print(f"Servidor {nome} está pronto.")
                return
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.ReadTimeout:
            print(f"Tentativa de conexão com {nome} expirou, tentando novamente...")
        time.sleep(0.5)

# 1. Inicia o servidor Python em uma thread separada
thread_servidor = threading.Thread(target=iniciar_servidor_python, daemon=True)
thread_servidor.start()

# 2. Aguarda o servidor Python (porta 5000)
aguardar_servidor("http://localhost:5000", "Python")

# 3. Aguarda o servidor Java (porta 8080)
print("\nAVISO: Certifique-se de ter iniciado o servidor Java (ex: 'mvn spring-boot:run')")
aguardar_servidor("http://localhost:8080", "Java")

print("\n=== AMBOS OS SERVIDORES ESTÃO PRONTOS. INICIANDO SIMULAÇÃO. ===\n")

# 4. Envia sinais para ambos os servidores
client.simula_clientes()