import os
from pathlib import Path
from PIL import Image

def polygon_to_yolo_format(label_txt_path, img_path, class_to_id):
    """
    Converte arquivo txt de polígono para formato YOLO.
    label_txt_path: Path do label txt original
    img_path: Path da imagem correspondente
    class_to_id: dict {class_name: id}
    Retorna: lista de strings no formato YOLO
    """
    with open(label_txt_path, 'r') as f:
        lines = f.read().strip().split('\n')

    # Pega largura e altura da imagem
    with Image.open(img_path) as img:
        img_w, img_h = img.size

    yolo_labels = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 10:
            print(f"Aviso: linha com formato inesperado em {label_txt_path}: {line}")
            continue

        # Pega os 8 valores das coordenadas x,y (4 pontos)
        coords = list(map(float, parts[:8]))
        # Pega o nome da classe (penúltimo valor)
        class_name = parts[8]
        # Ignorar o último valor (índice, opcional)

        xs = coords[0::2]
        ys = coords[1::2]

        # Calcula bbox mínima que envolve o polígono
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        # Centro e tamanho
        cx = (x_min + x_max) / 2 / img_w
        cy = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        # Mapeia nome da classe para id, adiciona no dict se novo
        if class_name not in class_to_id:
            class_to_id[class_name] = len(class_to_id)

        class_id = class_to_id[class_name]

        yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_labels


def convert_dataset_labels(dataset_root, output_root=None):
    """
    Converte labels do formato polígono para YOLO e salva.
    dataset_root: pasta raiz do dataset contendo 'images' e 'labelTxt'
    output_root: pasta onde salvar os labels convertidos (se None, sobrescreve os originais)
    """
    dataset_root = Path(dataset_root)
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labelTxt"

    if output_root is None:
        output_root = labels_root
    else:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    class_to_id = {}

    for split in ['train', 'valid', 'test']:
        label_dir = labels_root / split
        image_dir = images_root / split

        if not label_dir.exists():
            print(f"Pasta de labels {label_dir} não existe, pulando.")
            continue
        if not image_dir.exists():
            print(f"Pasta de imagens {image_dir} não existe, pulando.")
            continue

        output_split_dir = output_root / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        label_files = list(label_dir.glob("*.txt"))
        print(f"Convertendo {len(label_files)} arquivos em {split}...")

        for label_file in label_files:
            img_name = label_file.stem + ".png"  # assume .png, ajuste se necessário
            img_path = image_dir / img_name

            if not img_path.exists():
                print(f"Imagem {img_path} não encontrada, pulando label {label_file}")
                continue

            yolo_labels = polygon_to_yolo_format(label_file, img_path, class_to_id)

            # Salva novo label
            out_file = output_split_dir / label_file.name
            with open(out_file, 'w') as f:
                f.write('\n'.join(yolo_labels))

    print("Mapeamento de classes:")
    for k,v in class_to_id.items():
        print(f"{v}: {k}")

    print("Conversão finalizada.")

if __name__ == "__main__":
    dataset_path = "/home/jonasvm/docker-images/dota_dataset"  # caminho pro seu dataset raiz
    convert_dataset_labels(dataset_path)
