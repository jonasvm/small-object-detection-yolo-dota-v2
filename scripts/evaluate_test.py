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
INFER_RUNS = 30   # iterações para medir latência/FPS/uso de recursos

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
    import thop  # noqa
    from thop import profile as thop_profile
    thop = thop_profile
except Exception:
    pass


# ======== Helpers ========
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
    """
    Tenta extrair métricas por tamanho (small/medium/large).
    Retorna dict com chaves: map50_s, map50_m, map50_l, map_s, map_m, map_l.
    Se a versão não expõe, devolve NaN.
    """
    out = {
        'map50_s': np.nan, 'map50_m': np.nan, 'map50_l': np.nan,
        'map_s':   np.nan, 'map_m':   np.nan, 'map_l':   np.nan,
    }
    if m is None or not hasattr(m, 'box'):
        return out
    box = m.box
    # Tentativas comuns em versões do Ultralytics/COCO evaluator
    for k_src, k_dst in [
        ('map50_small', 'map50_s'), ('map50_s', 'map50_s'),
        ('map50_medium', 'map50_m'), ('map50_m', 'map50_m'),
        ('map50_large', 'map50_l'), ('map50_l', 'map50_l'),
        ('map_small', 'map_s'), ('maps', 'map_s'),  # cuidado: 'maps' geralmente é por CLASSE, não por tamanho
        ('map_medium', 'map_m'),
        ('map_large', 'map_l'),
    ]:
        val = getattr(box, k_src, None)
        if isinstance(val, (float, int)):
            out[k_dst] = float(val)
    return out

def extract_ap_per_class(m):
    """
    Retorna (classes, ap_per_class) se disponível.
    Em muitas versões, metrics.box.maps é array de AP (IoU 0.5:0.95) por classe.
    """
    if m is None or not hasattr(m, 'box'):
        return [], []
    maps = getattr(m.box, 'maps', None)
    if maps is None:
        return [], []
    try:
        ap = [float(x) for x in maps]
    except Exception:
        return [], []
    # tentar nomes de classes via m.names
    names = getattr(m, 'names', None)
    if not names:
        # names pode estar no próprio model
        return list(range(len(ap))), ap
    # names pode ser dict {i: 'name'}
    if isinstance(names, dict):
        ordered = [names.get(i, str(i)) for i in range(len(ap))]
        return ordered, ap
    # ou lista
    if isinstance(names, list):
        return names[:len(ap)], ap
    return list(range(len(ap))), ap

def eval_split(model, data_yaml, split):
    """Avalia split via model.val(split=...)."""
    return model.val(data=data_yaml, imgsz=IMG_SZ, split=split, verbose=False)

def measure_latency_resources(model, imgsz=IMG_SZ, runs=INFER_RUNS, device='cpu'):
    """
    Mede latência média (ms), FPS, RAM/CPU médios & picos, e energia GPU (Joules) se disponível.
    """
    # imagem dummy
    img = (np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8))

    # Inicializa psutil
    proc = psutil.Process(os.getpid()) if psutil else None

    # Inicializa NVML
    gpu_energy_j = np.nan
    gpu_power_samples = []
    gpu_power_times = []

    if pynvml:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            handle = None
    else:
        handle = None

    # Warm-up
    for _ in range(3):
        _ = model.predict(img, device=device, imgsz=imgsz, verbose=False)

    times = []
    cpu_perc = []
    ram_mb = []
    ram_peak_mb = 0.0

    t_energy_start = time.perf_counter()
    for _ in range(runs):
        # coleta pré
        if proc:
            try:
                ram_now = proc.memory_info().rss / (1024**2)
                ram_peak_mb = max(ram_peak_mb, ram_now)
                ram_mb.append(ram_now)
                cpu_perc.append(proc.cpu_percent(interval=None))  # medição rápida
            except Exception:
                pass

        if handle:
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                gpu_power_samples.append(p)
                gpu_power_times.append(time.perf_counter())
            except Exception:
                pass

        t0 = time.perf_counter()
        _ = model.predict(img, device=device, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()

        times.append(t1 - t0)

        if handle:
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                gpu_power_samples.append(p)
                gpu_power_times.append(time.perf_counter())
            except Exception:
                pass

    t_energy_end = time.perf_counter()

    latency_ms = (sum(times) / len(times)) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else np.nan

    cpu_avg = float(np.mean(cpu_perc)) if cpu_perc else np.nan
    ram_avg = float(np.mean(ram_mb)) if ram_mb else np.nan

    # Integra potência para energia (trapézio) — GPU
    if handle and len(gpu_power_samples) >= 2:
        E = 0.0
        for i in range(1, len(gpu_power_samples)):
            dt = gpu_power_times[i] - gpu_power_times[i-1]
            E += 0.5 * (gpu_power_samples[i] + gpu_power_samples[i-1]) * dt  # Joules
        gpu_energy_j = E
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    total_infer_time_s = t_energy_end - t_energy_start

    return {
        'latency_ms': latency_ms,
        'fps': fps,
        'cpu_avg_percent': cpu_avg,
        'ram_avg_mb': ram_avg,
        'ram_peak_mb': ram_peak_mb,
        'gpu_energy_j': gpu_energy_j,
        'infer_wall_time_s': total_infer_time_s,
    }

def get_model_complexity(yolo_model, imgsz=IMG_SZ):
    """
    Tenta obter #parâmetros e FLOPs.
    1) via thop (se instalado)
    2) via model.info() (parse do texto)
    """
    # 1) THOP
    if thop:
        try:
            import torch
            dummy = torch.zeros(1, 3, imgsz, imgsz)
            m = yolo_model.model  # nn.Module
            flops, params = thop(m, (dummy,), verbose=False)
            # THOP retorna MACs; alguns tratam como FLOPs ≈ 2*MACs — aqui deixamos como MACs->FLOPs*2
            gflops = (flops * 2) / 1e9
            n_params = params
            return float(n_params), float(gflops)
        except Exception:
            pass

    # 2) Parse do model.info()
    try:
        from io import StringIO
        import contextlib
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            yolo_model.info(detailed=False, verbose=True, imgsz=imgsz)
        txt = buf.getvalue()

        # Procura por números de Params e GFLOPs no texto
        # Exemplos comuns: "Params: 11.2M" / "GFLOPs: 6.3"
        params = np.nan
        flops = np.nan

        m1 = re.search(r'Params:\s*([\d\.]+)\s*([MkB])', txt, re.IGNORECASE)
        if m1:
            val, unit = float(m1.group(1)), m1.group(2).lower()
            mult = 1.0
            if unit == 'm': mult = 1e6
            elif unit == 'k': mult = 1e3
            elif unit == 'b': mult = 1e9
            params = val * mult

        m2 = re.search(r'GFLOPs:\s*([\d\.]+)', txt, re.IGNORECASE)
        if m2:
            flops = float(m2.group(1))

        return float(params), float(flops)
    except Exception:
        return np.nan, np.nan

def read_total_training_time(results_csv_path):
    """
    Tenta obter tempo total de treino do results.csv do Ultralytics.
    Procura colunas típicas: 'time', 'epoch_time', 'elapsed', etc.
    Se não achar, retorna NaN.
    """
    try:
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()
        # Possíveis colunas de tempo:
        for col in ['time', 'epoch_time', 'elapsed', 'train/elapsed', 'epoch/elapsed']:
            if col in df.columns:
                tot = float(df[col].sum())
                return tot
        # às vezes há 'epoch' + 'minutes' no log — difícil inferir; fallback:
        return np.nan
    except Exception:
        return np.nan


# ======== Pipeline ========
model = YOLO(model_path)

# Avaliações por split
metrics_train = eval_split(model, data_yaml, 'train')
metrics_val   = eval_split(model, data_yaml, 'val')
metrics_test  = eval_split(model, data_yaml, 'test')

# Básicas
map50_tr, map5095_tr, prec_tr, rec_tr = extract_basic_metrics(metrics_train)
map50_va, map5095_va, prec_va, rec_va = extract_basic_metrics(metrics_val)
map50_te, map5095_te, prec_te, rec_te = extract_basic_metrics(metrics_test)

# Por tamanho
size_tr = extract_size_metrics(metrics_train)
size_va = extract_size_metrics(metrics_val)
size_te = extract_size_metrics(metrics_test)

# AP por classe (do split de validação, por ex.)
cls_names, ap_per_class = extract_ap_per_class(metrics_val)
if cls_names and ap_per_class:
    ap_df = pd.DataFrame({'class': cls_names, 'AP@0.5:0.95(val)': ap_per_class})
    ap_csv_path = os.path.join(os.path.dirname(model_path), 'ap_per_class_val.csv')
    ap_df.to_csv(ap_csv_path, index=False)
else:
    ap_csv_path = None

# Losses do treino
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

# Tamanho do arquivo
model_size_mb = os.path.getsize(model_path) / 1e6

# Complexidade (#parâmetros e FLOPs)
n_params, gflops = get_model_complexity(model, imgsz=IMG_SZ)

# Latência/FPS + recursos/energia
resources = measure_latency_resources(model, imgsz=IMG_SZ, runs=INFER_RUNS, device='cpu')
latency_ms = resources['latency_ms']
fps        = resources['fps']
cpu_avg    = resources['cpu_avg_percent']
ram_avg    = resources['ram_avg_mb']
ram_peak   = resources['ram_peak_mb']
gpu_energy_j = resources['gpu_energy_j']
infer_wall  = resources['infer_wall_time_s']

# Tempo total de treino
total_train_time_s = read_total_training_time(results_csv)

# ======== Saídas ========
print("\n### 🧪 Resultados — YOLOv8n (extendido)")
print(
    "\n| Modelo | Seed | K-Fold | BBox | "
    "mAP@0.50(tr) | mAP@0.5:0.95(tr) | P(tr) | R(tr) | "
    "mAP@0.50(val) | mAP@0.5:0.95(val) | P(val) | R(val) | "
    "mAP@0.50(te) | mAP@0.5:0.95(te) | P(te) | R(te) | "
    "mAP50s/50m/50l(val) | mAPs/mAPm/mAPl(val) | "
    "FPS | Latência | FLOPs(G) | #Parâmetros | RAM(avg/peak MB) | CPU(%) | Energia GPU(J) | "
    "TrainTime(s) | Loss(tr) | Loss(val) | Tamanho |"
)
print(
    "|:------:|:----:|:------:|:----:|"
    ":--------:|:--------------:|:----:|:----:|"
    ":----------:|:----------------:|:------:|:-----:|"
    ":----------:|:----------------:|:------:|:-----:|"
    ":-----------------:|:-------------------:|"
    ":---:|:--------:|:-------:|:-----------:|:----------------:|:-----:|:-------------:|"
    ":-----------:|:-------:|:---------:|:-------:|"
)

def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and not math.isnan(x) else ("NaN" if isinstance(x, float) and math.isnan(x) else str(x))

print(
    f"| **YOLOv8n** | {SEED} | {KFOLD} | {BBOX} | "
    f"{fmt(map50_tr)} | {fmt(map5095_tr)} | {fmt(prec_tr)} | {fmt(rec_tr)} | "
    f"{fmt(map50_va)} | {fmt(map5095_va)} | {fmt(prec_va)} | {fmt(rec_va)} | "
    f"{fmt(map50_te)} | {fmt(map5095_te)} | {fmt(prec_te)} | {fmt(rec_te)} | "
    f"{fmt(size_va.get('map50_s'))}/{fmt(size_va.get('map50_m'))}/{fmt(size_va.get('map50_l'))} | "
    f"{fmt(size_va.get('map_s'))}/{fmt(size_va.get('map_m'))}/{fmt(size_va.get('map_l'))} | "
    f"{fmt(fps)} | {fmt(latency_ms)} ms | {fmt(gflops)} | {fmt(n_params)} | "
    f"{fmt(ram_avg)}/{fmt(ram_peak)} | {fmt(cpu_avg)} | {fmt(gpu_energy_j)} | "
    f"{fmt(total_train_time_s)} | {fmt(loss_train_final)} | {fmt(loss_val_final)} | {fmt(model_size_mb)} MB |"
)

# Extra: salvar AP por classe
if ap_csv_path:
    print(f"\n[OK] AP por classe (val) salvo em: {ap_csv_path}")
else:
    print("\n[INFO] AP por classe (val) não disponível nesta versão/execução.")

print("[OK] Métricas estendidas calculadas.")
