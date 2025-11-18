from ultralytics.data.converter import convert_dota_to_yolo_obb
import os

# Caminho para o dataset DOTA
dota_path = "/home/jonasvm/docker-images/dota_dataset_v15_hbb"

# Verifica se o diretório existe
if not os.path.isdir(dota_path):
    raise FileNotFoundError(f"O diretório '{dota_path}' não foi encontrado.")

# Converte para formato YOLO-OBB
print(f"Iniciando conversão do dataset DOTA em: {dota_path}")
convert_dota_to_yolo_obb(dota_path)
print("Conversão concluída com sucesso!")
