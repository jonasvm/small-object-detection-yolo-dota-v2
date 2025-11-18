import os
import re
import time
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ======== Config ========
model_path = '/home/jonasvm/docker-images/meus_resultados/detect/train45/weights/best.pt'
results_csv = '/home/jonasvm/docker-images/meus_resultados/detect/train45/results.csv'
data_yaml   = '/home/jonasvm/docker-images/dota_dataset_v15_hbb/dataset_config_fold0.yaml'

SEED  = 42
KFOLD = 0
BBOX  = 'HBB'
IMG_SZ = 1024
INFER_RUNS = 30

# ======== Imports opcionais ========
psutil = None
try:
    import psutil as _psutil
    psutil = _psutil
except Exception:
    pass

pynvml = None
try:
    import pynvml as _pynvml
    pynvml = _pynvml
except Exception:
    pass

thop = None
try:
    import thop
    from thop import profile as thop_profile
    thop = thop_profile
except Exception:
    pass


# ======== Funções utilitárias ========
def extract_basic_metrics(m):
    """Retorna (map50, map5095, precision, recall) ou NaNs."""
    if m is None or not hasattr(m, 'box'):
        return (np.nan, np.nan, np.nan, np.nan)
    return (
        float(getattr(m.box, 'map50', np.nan)),
        float(getattr(m.box, 'map',   np.nan)),
        float(getattr(m.box, 'mp',    np.nan)),
        float(getattr(m.box, 'mr',    np.nan)),
    )

def extract_size_metrics(m):
    """Tenta extrair métricas por tamanho (small/medium/large)."""
    out = {'map50_s': np.nan, 'map50_m': np.nan, 'map50_l': np.nan,
           'map_s': np.nan, 'map_m': np.nan, 'map_l': np.nan}
    if m is None or not hasattr(m, 'box'):
        return out
    box = m.box
    for src, dst in [
        ('map50_small', 'map50_s'), ('map50_s', 'map50_s'),
        ('map50_medium', 'map50_m'), ('map50_m', 'map50_m'),
        ('map50_large', 'map50_l'), ('map50_l', 'map50_l'),
        ('map_small', 'map_s'), ('map_medium', 'map_m'), ('map_large', 'map_l')
    ]:
        val = getattr(box, src, None)
        if isinstance(val, (float, int)):
            out[dst] = float(val)
    return out

def extract_ap_per_class(m):
    """Retorna listas (nomes, AP por classe)."""
    if m is None or not hasattr(m, 'box'):
        return [], []
    maps = getattr(m.box, 'maps', None)
    if maps is None:
        return [], []
    ap = [float(x) for x in maps]
    names = getattr(m, 'names', None)
    if isinstance(names, dict):
        names = [names.get(i, str(i)) for i in range(len(ap))]
    elif not isinstance(names, list):
        names = list(range(len(ap)))
    return names, ap

def eval_split(model, data_yaml, split):
    """Avalia split via model.val(split=...)."""
    return model.val(data=data_yaml, imgsz=IMG_SZ, split=split, verbose=False)

def measure_latency_resources(model, imgsz=IMG_SZ, runs=INFER_RUNS, device='cpu'):
    """Mede latência média (ms), FPS, RAM/CPU médios & picos, energia GPU (J)."""
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    proc = psutil.Process(os.getpid()) if psutil else None

    handle = None
    if pynvml:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            handle = None

    # Warm-up
    for _ in range(3):
        _ = model.predict(img, device=device, imgsz=imgsz, verbose=False)

    times, cpu_use, ram_use = [], [], []
    ram_peak = 0.0
    power_samples, time_samples = [], []

    for _ in range(runs):
        if proc:
            ram_now = proc.memory_info().rss / (1024**2)
            ram_peak = max(ram_peak, ram_now)
            ram_use.append(ram_now)
            cpu_use.append(proc.cpu_percent(interval=None))
        if handle:
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_samples.append(p)
                time_samples.append(time.perf_counter())
            except Exception:
                pass
        t0 = time.perf_counter()
        _ = model.predict(img, device=device, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    latency_ms = (sum(times) / len(times)) * 1000
    fps = 1000 / latency_ms if latency_ms > 0 else np.nan
    cpu_avg = np.mean(cpu_use) if cpu_use else np.nan
    ram_avg = np.mean(ram_use) if ram_use else np.nan
    gpu_energy = np.nan
    if handle and len(power_samples) > 1:
        E = 0
        for i in range(1, len(power_samples)):
            dt = time_samples[i] - time_samples[i-1]
            E += 0.5 * (power_samples[i] + power_samples[i-1]) * dt
        gpu_energy = E
        pynvml.nvmlShutdown()
    return dict(latency_ms=latency_ms, fps=fps, cpu=cpu_avg, ram=ram_avg,
                ram_peak=ram_peak, gpu_energy=gpu_energy)

def get_model_complexity(model, imgsz=IMG_SZ):
    """Tenta obter #parâmetros e GFLOPs."""
    if thop:
        import torch
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        flops, params = thop(model.model, (dummy,), verbose=False)
        return float(params), float(flops * 2 / 1e9)
    try:
        from io import StringIO
        import contextlib
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            model.info(detailed=False, verbose=True, imgsz=imgsz)
        txt = buf.getvalue()
        params = re.search(r'Params:\s*([\d\.]+)\s*([MkB])', txt)
        flops  = re.search(r'GFLOPs:\s*([\d\.]+)', txt)
        n_p = float(params.group(1)) * {'m':1e6,'k':1e3,'b':1e9}.get(params.group(2).lower(),1) if params else np.nan
        f_g = float(flops.group(1)) if flops else np.nan
        return n_p, f_g
    except Exception:
        return np.nan, np.nan

def read_total_training_time(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for c in ['time','epoch_time','elapsed','train/elapsed']:
            if c in df.columns:
                return float(df[c].sum())
        return np.nan
    except Exception:
        return np.nan

# ======== Avaliação ========
model = YOLO(model_path)

metrics_train = eval_split(model, data_yaml, 'train')
metrics_val   = eval_split(model, data_yaml, 'val')
metrics_test  = eval_split(model, data_yaml, 'test')

map50_tr, map5095_tr, prec_tr, rec_tr = extract_basic_metrics(metrics_train)
map50_va, map5095_va, prec_va, rec_va = extract_basic_metrics(metrics_val)
map50_te, map5095_te, prec_te, rec_te = extract_basic_metrics(metrics_test)

size_va = extract_size_metrics(metrics_val)
cls_names, ap_per_class = extract_ap_per_class(metrics_val)

df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()
cols_train = [c for c in df.columns if 'train' in c and 'loss' in c]
cols_val   = [c for c in df.columns if 'val' in c and 'loss' in c]
df['train_loss'] = df[cols_train].sum(axis=1)
df['val_loss']   = df[cols_val].sum(axis=1)
loss_train_final = df['train_loss'].iloc[-1]
loss_val_final   = df['val_loss'].iloc[-1]

model_size_mb = os.path.getsize(model_path) / 1e6
n_params, gflops = get_model_complexity(model, imgsz=IMG_SZ)
resources = measure_latency_resources(model, imgsz=IMG_SZ)
total_train_time_s = read_total_training_time(results_csv)

# ======== Impressão ========
print("\n### 🧪 Resultados — YOLOv8n (completo no terminal)\n")
print(f"📁 Modelo: {model_path}")
print(f"🔢 Seed={SEED} | K-Fold={KFOLD} | BBox={BBOX} | IMG={IMG_SZ}px\n")

print("🔹 Métricas gerais:")
print(f"  Train:  mAP@50={map50_tr:.3f}, mAP@0.5:0.95={map5095_tr:.3f}, Prec={prec_tr:.3f}, Rec={rec_tr:.3f}")
print(f"  Val:    mAP@50={map50_va:.3f}, mAP@0.5:0.95={map5095_va:.3f}, Prec={prec_va:.3f}, Rec={rec_va:.3f}")
print(f"  Test:   mAP@50={map50_te:.3f}, mAP@0.5:0.95={map5095_te:.3f}, Prec={prec_te:.3f}, Rec={rec_te:.3f}")

print("\n🔹 Métricas por tamanho (val):")
for k, v in size_va.items():
    print(f"  {k:10s}: {v if not np.isnan(v) else 'NaN'}")

print("\n🔹 Recursos e desempenho:")
print(f"  Latência média (CPU): {resources['latency_ms']:.1f} ms")
print(f"  FPS estimado: {resources['fps']:.1f}")
print(f"  RAM média: {resources['ram']:.1f} MB (pico {resources['ram_peak']:.1f} MB)")
print(f"  CPU média: {resources['cpu']:.1f}%")
print(f"  Energia GPU estimada: {resources['gpu_energy']:.2f} J")
print(f"  FLOPs: {gflops:.3f} GFLOPs | Parâmetros: {n_params/1e6:.2f} M")
print(f"  Tempo total de treino: {total_train_time_s:.1f} s")
print(f"  Tamanho do modelo: {model_size_mb:.1f} MB")

print("\n🔹 Losses finais:")
print(f"  Train Loss: {loss_train_final:.3f} | Val Loss: {loss_val_final:.3f}")

if cls_names and ap_per_class:
    ap_df = pd.DataFrame({'Classe': cls_names, 'AP@0.5:0.95 (val)': ap_per_class})
    print("\n📊 AP por classe (val):")
    print(ap_df.to_string(index=False))
else:
    print("\n[INFO] AP por classe (val) não disponível nesta versão/execução.")

print("\n✅ Avaliação concluída com sucesso — todos os resultados exibidos no terminal.")
