import os
from pathlib import Path

# Caminhos
IMG_EXT = ('.png', '.jpg', '.jpeg')
IMG_DIR = Path("home/jonasvm/docker-images/dota_dataset/images/val")  # ou val, se quiser mudar
LBL_DIR = Path("home/jonasvm/docker-images/dota_dataset/labelTxt/val")  # ou val

# Coletar imagens e labels
img_files = sorted([f.stem for f in IMG_DIR.glob("*") if f.suffix.lower() in IMG_EXT])
lbl_files = sorted([f.stem for f in LBL_DIR.glob("*.txt")])

# Verificações
sem_label = [f for f in img_files if f not in lbl_files]
sem_imagem = [f for f in lbl_files if f not in img_files]

# Verificar arquivos de label vazios (ou com apenas cabeçalho)
vazios = []
for lbl_name in lbl_files:
    lbl_path = LBL_DIR / f"{lbl_name}.txt"
    with open(lbl_path, 'r') as f:
        linhas = f.readlines()
        if len(linhas) <= 2:
            vazios.append(lbl_name)

# Resultados
print("==== RESULTADO ====")
print(f"Total de imagens       : {len(img_files)}")
print(f"Total de labels        : {len(lbl_files)}")
print(f"Imagens sem label      : {len(sem_label)}")
print(f"Labels sem imagem      : {len(sem_imagem)}")
print(f"Labels vazios/inválidos: {len(vazios)}")

# Exibir exemplos (limitado a 10)
if sem_label:
    print("\nExemplo de imagens sem label:")
    print("\n".join(sem_label[:10]))

if sem_imagem:
    print("\nExemplo de labels sem imagem:")
    print("\n".join(sem_imagem[:10]))

if vazios:
    print("\nExemplo de labels vazios:")
    print("\n".join(vazios[:10]))
