import random

MODELS = ['H_60x60.csv', 'H_30x30.csv']
SIGNAL60 = ['sinal_1_60x60.csv', 'sinal_2_60x60.csv', 'sinal_3_60x60.csv']
SIGNAL30 = ['sinal_1_30x30.csv', 'sinal_2_30x30.csv', 'sinal_3_30x30.csv']
ALGORITHM = ['CGNE', 'CGNR'] 
GAIN_OPTIONS = [True, False]

def perform_sorteio(num_sinais, output_filename='requisicoes.txt'):
    final_output_filename = f"sorteio_{output_filename}"
    
    print(f"Gerando arquivo de sorteio: {final_output_filename} com {num_sinais} entradas...")

    with open(final_output_filename, 'w') as f:
        for i in range(num_sinais):
            model = random.choice(MODELS)
            
            if model == 'H_60x60.csv':
                signal = random.choice(SIGNAL60)
            else: 
                signal = random.choice(SIGNAL30)
                
            algorithm = random.choice(ALGORITHM)
            
            has_gain = random.choice(GAIN_OPTIONS)
            
            f.write(f"{model},{signal},{algorithm},{has_gain}\n")
            
    print(f"Sorteio concluído. Arquivo salvo em: {final_output_filename}")
    return final_output_filename

if __name__ == "__main__":
    perform_sorteio(num_sinais=30)