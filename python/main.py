import client
import server
import time
import threading
from waitress import serve
import requests

def iniciar_servidor():
    serve(server.app, host='0.0.0.0', port=5000, threads = 5)

#Função pra fazer os clientes aguardar o servidor iniciar antes de tentarem se conectar
def aguardar_servidor(url):
    
    while True:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("Servidor está pronto.")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)


# Inicia o servidor em uma thread separada (precisa pois o app.run é bloqueante )
thread_servidor = threading.Thread(target=iniciar_servidor, daemon=True)
thread_servidor.start()

time.sleep(1)

# Aguarda o servidor iniciar
aguardar_servidor("http://localhost:5000/ping")

# Envia sinais
client.simula_clientes()