import os
import shutil
from pathlib import Path
from sklearn.model_selection import KFold
import yaml

# === CONFIGURAÇÃO ===
orig_dataset_yaml = Path('/home/jonasvm/docker-images/dota_dataset_v15_hbb/dataset_config.yaml')
saida_base = Path('/home/jonasvm/docker-images/dota_dataset_v15_hbb/kfold')
orig_image_dir = Path('/home/jonasvm/docker-images/dota_dataset_v15_hbb/images/train')
orig_label_dir = Path('/home/jonasvm/docker-images/dota_dataset_v15_hbb/labels/train')
k = 3

# === CARREGAR CONFIG BASE ===
with orig_dataset_yaml.open() as f:
    base_config = yaml.safe_load(f)

nc = base_config['nc']
names = base_config['names']
test_path = base_config.get('test', '/home/jonasvm/docker-images/dota_dataset_v15_hbb/images/test')

# === COLETAR ARQUIVOS ===
image_paths = sorted(orig_image_dir.glob('*.[jp][pn]g'))  # .jpg ou .png
label_paths = [orig_label_dir / (img.stem + '.txt') for img in image_paths]

assert all(l.exists() for l in label_paths), 'Algum arquivo de label está faltando.'

kf = KFold(n_splits=k, shuffle=True, random_state=42)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f'🧩 Gerando fold {fold_idx}...')

    fold_dir = saida_base / f'fold_{fold_idx}'
    paths = {
        'train_images': fold_dir / 'images/train',
        'val_images': fold_dir / 'images/val',
        'train_labels': fold_dir / 'labels/train',
        'val_labels': fold_dir / 'labels/val',
    }

    # Criar diretórios
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    # Copiar arquivos de treino
    for idx in train_idx:
        shutil.copy(image_paths[idx], paths['train_images'] / image_paths[idx].name)
        shutil.copy(label_paths[idx], paths['train_labels'] / label_paths[idx].name)

    # Copiar arquivos de validação
    for idx in val_idx:
        shutil.copy(image_paths[idx], paths['val_images'] / image_paths[idx].name)
        shutil.copy(label_paths[idx], paths['val_labels'] / label_paths[idx].name)

    # Criar dataset.yaml para o fold
    dataset_yaml = {
        'train': str(paths['train_images']),
        'val': str(paths['val_images']),
        'test': test_path,
        'nc': nc,
        'names': names
    }

    with open(fold_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    print(f'✅ Fold {fold_idx} gerado com sucesso.')

print('\n🎉 Todos os folds foram criados em:', saida_base)
