import os

label_dir = '/home/jonasvm/docker-images/dota_dataset_v15_hbb/labels/train_original'  # ajuste o caminho se necessário

for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(label_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Filtra só as linhas que começam com número (linha de caixa)
        cleaned_lines = [line for line in lines if line and line[0].isdigit()]
        
        with open(filepath, 'w') as f:
            f.writelines(cleaned_lines)

print("Arquivos de label limpos dos cabeçalhos extras!")
