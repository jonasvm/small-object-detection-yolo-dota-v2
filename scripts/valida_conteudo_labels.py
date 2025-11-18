import os
from pathlib import Path

LABEL_DIR = Path("home/jonasvm/docker-images/dota_dataset/labelTxt/val")  # ajuste para val se quiser
CLASSES_VALIDAS = set(range(15))  # nc = 15

arquivos_invalidos = []

for file in LABEL_DIR.glob("*.txt"):
    with open(file, 'r') as f:
        linhas = f.readlines()[2:]  # pular cabeçalho

        for i, linha in enumerate(linhas, start=3):
            partes = linha.strip().split()
            if len(partes) != 10:
                arquivos_invalidos.append((file.name, i, "colunas inválidas"))
                continue

            try:
                coords = list(map(float, partes[:8]))
                classe = int(partes[8])
                dificuldade = int(partes[9])
            except ValueError:
                arquivos_invalidos.append((file.name, i, "valores não numéricos"))
                continue

            if classe not in CLASSES_VALIDAS:
                arquivos_invalidos.append((file.name, i, f"classe inválida: {classe}"))

# Resultado
if not arquivos_invalidos:
    print("✅ Nenhum erro encontrado nos labels!")
else:
    print(f"⚠️ {len(arquivos_invalidos)} problemas encontrados:")
    for fname, linha, erro in arquivos_invalidos[:20]:  # mostrar os primeiros 20
        print(f"{fname} - linha {linha}: {erro}")
