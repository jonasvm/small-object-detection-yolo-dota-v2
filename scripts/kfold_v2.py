import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
import yaml

# === CONFIG ===
base_dir = Path('/home/jonasvm/docker-images/dota_dataset_v15_hbb')
orig_dataset_yaml = base_dir / 'dataset_config.yaml'
saida_base = base_dir / 'kfold'
image_dirs = [base_dir / 'images/train', base_dir / 'images/val']
label_dirs = [base_dir / 'labels/train', base_dir / 'labels/val']
k = 3
test_size = 0.15
random_state = 21

# === CARREGAR CONFIG BASE ===
with orig_dataset_yaml.open() as f:
    base_config = yaml.safe_load(f)

nc = base_config['nc']
names = base_config['names']

# === JUNTAR TODAS AS IMAGENS E LABELS DE TRAIN E VAL  ===
image_paths = []
for image_dir in image_dirs:
    image_paths += sorted(image_dir.glob('*.[jp][pn]g'))  # .jpg ou .png

label_paths = [Path(str(p).replace('/images/', '/labels/')).with_suffix('.txt') for p in image_paths]
assert all(l.exists() for l in label_paths), 'Algum arquivo de label está faltando.'

# === SEPARAR 15% PARA TESTE ===
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    image_paths, label_paths, test_size=test_size, random_state=random_state, shuffle=True
)

# === SALVAR TESTE EM PASTA FIXA ===
test_image_dir = saida_base / 'test' / 'images'
test_label_dir = saida_base / 'test' / 'labels'

for d in [test_image_dir, test_label_dir]:
    d.mkdir(parents=True, exist_ok=True)

for img, lbl in zip(test_imgs, test_labels):
    shutil.copy(img, test_image_dir / img.name)
    shutil.copy(lbl, test_label_dir / lbl.name)

# === FAZER KFOLD COM O RESTANTE ===
kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_imgs)):
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
        img, lbl = train_imgs[idx], train_labels[idx]
        shutil.copy(img, paths['train_images'] / img.name)
        shutil.copy(lbl, paths['train_labels'] / lbl.name)

    # Copiar arquivos de validação
    for idx in val_idx:
        img, lbl = train_imgs[idx], train_labels[idx]
        shutil.copy(img, paths['val_images'] / img.name)
        shutil.copy(lbl, paths['val_labels'] / lbl.name)

    # === Copiar dados de teste para dentro do fold ===
    fold_test_image_dir = fold_dir / 'images/test'
    fold_test_label_dir = fold_dir / 'labels/test'

    for d in [fold_test_image_dir, fold_test_label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for img in test_image_dir.glob('*'):
        shutil.copy(img, fold_test_image_dir / img.name)

    for lbl in test_label_dir.glob('*'):
        shutil.copy(lbl, fold_test_label_dir / lbl.name)

    # === Criar dataset.yaml para o fold ===
    dataset_yaml = {
        'train': str(paths['train_images']),
        'val': str(paths['val_images']),
        'test': str(fold_test_image_dir),
        'nc': nc,
        'names': names
    }

    with open(fold_dir / 'dataset_config.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    print(f'✅ Fold {fold_idx} gerado com sucesso.')

print('\n🎉 Todos os folds foram criados em:', saida_base)
