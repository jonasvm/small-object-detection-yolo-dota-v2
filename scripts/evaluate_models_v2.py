import os
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ===== Caminhos (Linux) =====
model_path = '/home/jonasvm/docker-images/meus_resultados/detect/train28/weights/best.pt'
results_csv = '/home/jonasvm/docker-images/meus_resultados/detect/train28/results.csv'
data_yaml  = '/home/jonasvm/docker-images/dota_dataset_v15_hbb/dataset_config_val.yaml'

# ===== Util =====
def extract_metrics(m):
    """Extrai métricas de detecção (retorna NaN se não existir)."""
    import numpy as _np
    if m is None or not hasattr(m, 'box'):
        return _np.nan, _np.nan, _np.nan, _np.nan
    return (
        float(getattr(m.box, 'map50', _np.nan)),
        float(getattr(m.box, 'map',   _np.nan)),
        float(getattr(m.box, 'mp',    _np.nan)),
        float(getattr(m.box, 'mr',    _np.nan)),
    )

# ===== Carregar modelo =====
model = YOLO(model_path)

# ===== Avaliações =====
# Treino (usa split='train' do YAML)
metrics_train = model.val(data=data_yaml, imgsz=1024, split='train', verbose=False)

# Validação (split default = 'val')
metrics_val   = model.val(data=data_yaml, imgsz=1024, verbose=False)

# Teste (requer campo test: no YAML). Use .test() para ser explícito.
metrics_test  = model.test(data=data_yaml, imgsz=1024, verbose=False)

# ===== Extrair métricas =====
map50_tr, map5095_tr, prec_tr, rec_tr = extract_metrics(metrics_train)
map50_va, map5095_va, prec_va, rec_va = extract_metrics(metrics_val)
map50_te, map5095_te, prec_te, rec_te = extract_metrics(metrics_test)

# ===== Ler CSV para losses (do treino anterior) =====
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()

cols_train = [c for c in df.columns if c.startswith('train/') and c.endswith('_loss')]
cols_val   = [c for c in df.columns if c.startswith('val/')   and c.endswith('_loss')]

if not cols_train or not cols_val:
    cols_train = [c for c in df.columns if 'train' in c and 'loss' in c]
    cols_val   = [c for c in df.columns if 'val'   in c and 'loss' in c]

df['train_loss'] = df[cols_train].sum(axis=1)
df['val_loss']   = df[cols_val].sum(axis=1)

loss_train_final = float(df['train_loss'].iloc[-1])
loss_val_final   = float(df['val_loss'].iloc[-1])

# ===== Tamanho do arquivo do modelo =====
model_size_mb = os.path.getsize(model_path) / 1e6

# ===== Latência média em CPU com imagem aleatória =====
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

# ===== Imprimir Tabela Markdown (clara: Treino vs Validação vs Teste) =====
print("\n### 🧪 Resultados — YOLOv11m (100 épocas, imgsz 1024, Adam, HBB)")
print("\n| Modelo    | Seed | K-Fold | Bounding Box | mAP@0.50 (train) | mAP@0.50:0.95 (train) | Precision (train) | Recall (train) | mAP@0.50 (val) | mAP@0.50:0.95 (val) | Precision (val) | Recall (val) | mAP@0.50 (test) | mAP@0.50:0.95 (test) | Precision (test) | Recall (test) | Latência (CPU) | Loss Final (train) | Loss Final (val) | Tamanho do Arquivo |")
print("|:----------|:----:|:------:|:------------:|:-----------------:|:----------------------:|:-----------------:|:--------------:|:--------------:|:--------------------:|:---------------:|:------------:|:---------------:|:---------------------:|:----------------:|:-------------:|:---------------:|:-------------------:|:----------------:|:------------------:|")
print(f"| **YOLOv11m** | 42 | 0 | HBB | {map50_tr:.3f} | {map5095_tr:.3f} | {prec_tr:.3f} | {rec_tr:.3f} | {map50_va:.3f} | {map5095_va:.3f} | {prec_va:.3f} | {rec_va:.3f} | {map50_te:.3f} | {map5095_te:.3f} | {prec_te:.3f} | {rec_te:.3f} | {latency_ms:.0f} ms | {loss_train_final:.3f} | {loss_val_final:.3f} | {model_size_mb:.1f} MB |")

# Também imprime um resumo simples no terminal (opcional)
print("\n[OK] Avaliações concluídas: train, val e test.")
