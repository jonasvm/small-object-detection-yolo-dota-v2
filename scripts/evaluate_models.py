import os
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ===== Caminhos (Linux) =====
model_path = '/home/jonasvm/docker-images/meus_resultados/detect/train28/weights/best.pt'
results_csv = '/home/jonasvm/docker-images/meus_resultados/detect/train28/results.csv'
data_yaml  = '/home/jonasvm/docker-images/dota_dataset_v15_hbb/dataset_config_val.yaml'

# ===== Carregar modelo =====
model = YOLO(model_path)

# ===== Validação (se quiser CPU, adicione device='cpu') =====
metrics = model.val(data=data_yaml, imgsz=1024, verbose=False)  # device='cpu' se desejar

# ===== Extrair métricas de performance =====
# (Algumas versões mudaram nomes; estes são os usuais no Ultralytics recente)
map50     = float(getattr(metrics.box, 'map50', np.nan))
map5095   = float(getattr(metrics.box, 'map',   np.nan))
precision = float(getattr(metrics.box, 'mp',    np.nan))
recall    = float(getattr(metrics.box, 'mr',    np.nan))

# ===== Ler CSV para losses =====
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()

cols_train = [c for c in df.columns if c.startswith('train/') and c.endswith('_loss')]
cols_val   = [c for c in df.columns if c.startswith('val/')   and c.endswith('_loss')]

if not cols_train or not cols_val:
    # Fallback simples caso o CSV tenha nomes diferentes
    # (ajuste aqui se o seu results.csv tiver outra convenção)
    cols_train = [c for c in df.columns if 'train' in c and 'loss' in c]
    cols_val   = [c for c in df.columns if 'val'   in c and 'loss' in c]

df['train_loss'] = df[cols_train].sum(axis=1)
df['val_loss']   = df[cols_val].sum(axis=1)

loss_train_final = float(df['train_loss'].iloc[-1])
loss_val_final   = float(df['val_loss'].iloc[-1])

# ===== Tamanho do arquivo do modelo =====
model_size_mb = os.path.getsize(model_path) / 1e6

# ===== Latência média em CPU com imagem aleatória =====
# Dica: mantenha imgsz consistente com o treino (1024 no seu caso)
imgsz = 1024
img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

# Warm-up (não medir)
for _ in range(3):
    _ = model.predict(img, device='cpu', imgsz=imgsz, verbose=False)

# Medição
runs = 20
times = []
for _ in range(runs):
    t0 = time.perf_counter()
    _ = model.predict(img, device='cpu', imgsz=imgsz, verbose=False)
    times.append(time.perf_counter() - t0)

latency_ms = (sum(times) / len(times)) * 1000.0

# ===== Imprimir tabela formatada =====
print(f'{"Modelo":<10} {"mAP @ 0.50":<12} {"mAP @ 0.50:0.95":<15} {"Precision":<9} {"Recall":<7} {"Latência (CPU)":<15} {"Loss Final (train)":<18} {"Loss Final (val)":<16} {"Arquivo":<8}')
print(f'{"yolov11":<10} {map50:<12.3f} {map5095:<15.3f} {precision:<9.3f} {recall:<7.3f} {latency_ms:>10.0f} ms   {loss_train_final:<18.3f} {loss_val_final:<16.3f} {model_size_mb:.1f} MB')
