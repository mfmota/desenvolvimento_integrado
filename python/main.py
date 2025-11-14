import client
import server
import time
import threading
from waitress import serve
import requests

def iniciar_servidor_python():
    print("Iniciando servidor Python na porta 5000...")
    serve(server.app, host='0.0.0.0', port=5000, threads=5)

def aguardar_servidor(url, nome):
    print(f"Aguardando servidor {nome} em {url}...")
    while True:
        try:
            r = requests.get(f"{url}/ping", timeout=1.0)
            if r.status_code == 200:
                print(f"Servidor {nome} está pronto.")
                return
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.ReadTimeout:
            print(f"Tentativa de conexão com {nome} expirou, tentando novamente...")
        time.sleep(0.5)

thread_servidor = threading.Thread(target=iniciar_servidor_python, daemon=True)
thread_servidor.start()

aguardar_servidor("http://localhost:5000", "Python")

print("\nAVISO: Certifique-se de ter iniciado o servidor Java (ex: 'mvn spring-boot:run')")
aguardar_servidor("http://localhost:8080", "Java")

print("\n=== AMBOS OS SERVIDORES ESTÃO PRONTOS. INICIANDO SIMULAÇÃO. ===\n")

client.simula_clientes()