import client
import time
from waitress import serve
import requests

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

aguardar_servidor("http://localhost:5000", "Python")

print("\nAVISO: Certifique-se de ter iniciado o servidor Java (ex: 'mvn spring-boot:run')")
aguardar_servidor("http://localhost:8080", "Java")

print("\n=== AMBOS OS SERVIDORES ESTÃO PRONTOS. INICIANDO SIMULAÇÃO. ===\n")

client.executar_cliente()