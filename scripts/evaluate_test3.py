#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ===================== CONFIG =====================
# Ajuste estes caminhos:
model_path = '/home/jonasvm/docker-images/meus_resultados/detect/train45/weights/best.pt'
results_csv = '/home/jonasvm/docker-images/meus_resultados/detect/train45/results.csv'
data_yaml   = '/home/jonasvm/docker-images/dota_dataset_v15_hbb/dataset_config_fold0.yaml'

# Metadados do experimento (para a linha Markdown)
MODEL_NAME = 'YOLOv8n'
SEED  = 42
KFOLD = 0
BBOX  = 'HBB'

# Outros parâmetros
IMG_SZ = 1024          # imgsz para validação e medição de latência
INFER_RUNS = 30        # iterações para medir latência/FPS/recursos
LAT_DEVICE = 'cpu'     # 'cpu' para medir latência em CPU; use 0,1,... para GPU se quiser medir na GPU

# ===================== IMPORTS OPCIONAIS =====================
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

thop_profile = None
try:
    from thop import profile as _thop_profile
    thop_profile = _thop_profile
except Exception:
    pass


# ===================== FUNÇÕES UTIL =====================
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
    """Retorna listas (nomes, AP por classe) usando metrics.box.maps se existir."""
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

def safe_eval_split(model, data_yaml, split):
    """
    Avalia via model.val(split=...). Se o split não existir no YAML/versão,
    retorna None e imprime aviso.
    """
    try:
        return model.val(data=data_yaml, imgsz=IMG_SZ, split=split, verbose=False)
    except Exception as e:
        print(f"[WARN] Falha ao avaliar split='{split}': {e}")
        return None

def measure_latency_resources(model, imgsz=IMG_SZ, runs=INFER_RUNS, device=LAT_DEVICE):
    """
    Mede latência média (ms), FPS, RAM/CPU médios & picos, e energia GPU (Joules) se disponível.
    Tudo exibido no terminal; não salva nada.
    """
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    proc = psutil.Process(os.getpid()) if psutil else None

    # NVML (GPU)
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
            try:
                ram_now = proc.memory_info().rss / (1024**2)
                ram_peak = max(ram_peak, ram_now)
                ram_use.append(ram_now)
                cpu_use.append(proc.cpu_percent(interval=None))
            except Exception:
                pass

        if handle:
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
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

    # Integra potência da GPU em energia (trapézio)
    if handle and len(power_samples) > 1:
        E = 0.0
        for i in range(1, len(power_samples)):
            dt = time_samples[i] - time_samples[i-1]
            E += 0.5 * (power_samples[i] + power_samples[i-1]) * dt
        gpu_energy = E
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return dict(latency_ms=latency_ms, fps=fps, cpu=cpu_avg, ram=ram_avg,
                ram_peak=ram_peak, gpu_energy=gpu_energy)

def get_model_complexity(yolo_model, imgsz=IMG_SZ):
    """
    Retorna (#parâmetros, GFLOPs) com robustez:
      1) THOP no mesmo device do modelo
      2) Fallback: THOP no CPU (cópia do modelo)
      3) Fallback: parse de model.info()
    """
    # 1) THOP no mesmo device
    if thop_profile:
        try:
            import torch
            m = yolo_model.model  # nn.Module interno
            m.eval()
            device = next(m.parameters()).device
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
            with torch.no_grad():
                flops, params = thop_profile(m, (dummy,), verbose=False)
            gflops = float(flops * 2 / 1e9)  # MACs -> FLOPs (~2x)
            return float(params), gflops
        except RuntimeError:
            # 2) THOP em CPU com cópia do modelo
            try:
                import torch, copy
                m_cpu = copy.deepcopy(yolo_model.model).cpu().eval()
                dummy = torch.zeros(1, 3, imgsz, imgsz, device='cpu')
                with torch.no_grad():
                    flops, params = thop_profile(m_cpu, (dummy,), verbose=False)
                gflops = float(flops * 2 / 1e9)
                return float(params), gflops
            except Exception:
                pass
        except Exception:
            pass

    # 3) Parse de model.info()
    try:
        from io import StringIO
        import contextlib
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            yolo_model.info(detailed=False, verbose=True, imgsz=imgsz)
        txt = buf.getvalue()

        params = np.nan
        flops  = np.nan

        m1 = re.search(r'Params:\s*([\d\.]+)\s*([MkB])', txt, re.IGNORECASE)
        if m1:
            val, unit = float(m1.group(1)), m1.group(2).lower()
            mult = {'m':1e6, 'k':1e3, 'b':1e9}.get(unit, 1.0)
            params = val * mult

        m2 = re.search(r'GFLOPs:\s*([\d\.]+)', txt, re.IGNORECASE)
        if m2:
            flops = float(m2.group(1))

        return float(params), float(flops)
    except Exception:
        return float('nan'), float('nan')

def read_total_training_time(csv_path):
    """Tenta somar tempos no results.csv (se existir colunas relevantes)."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for c in ['time', 'epoch_time', 'elapsed', 'train/elapsed', 'epoch/elapsed']:
            if c in df.columns:
                return float(df[c].sum())
        return np.nan
    except Exception:
        return np.nan

def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and not math.isnan(x) else ("NaN" if isinstance(x, float) and math.isnan(x) else str(x))


# ===================== PIPELINE =====================
model = YOLO(model_path)

# Avaliação por split (treino/val/test) — se algum split não existir, retorna None
metrics_train = safe_eval_split(model, data_yaml, 'train')
metrics_val   = safe_eval_split(model, data_yaml, 'val')
metrics_test  = safe_eval_split(model, data_yaml, 'test')

# Métricas gerais
map50_tr, map5095_tr, prec_tr, rec_tr = extract_basic_metrics(metrics_train)
map50_va, map5095_va, prec_va, rec_va = extract_basic_metrics(metrics_val)
map50_te, map5095_te, prec_te, rec_te = extract_basic_metrics(metrics_test)

# Métricas por tamanho (usando val como referência principal)
size_va = extract_size_metrics(metrics_val)

# AP por classe (val)
cls_names, ap_per_class = extract_ap_per_class(metrics_val)

# Losses finais do treino (somatório de colunas *_loss)
df_logs = pd.read_csv(results_csv)
df_logs.columns = df_logs.columns.str.strip()
cols_train = [c for c in df_logs.columns if 'train' in c and 'loss' in c]
cols_val   = [c for c in df_logs.columns if 'val' in c and 'loss' in c]
df_logs['train_loss'] = df_logs[cols_train].sum(axis=1) if cols_train else np.nan
df_logs['val_loss']   = df_logs[cols_val].sum(axis=1) if cols_val else np.nan
loss_train_final = float(df_logs['train_loss'].iloc[-1]) if 'train_loss' in df_logs else np.nan
loss_val_final   = float(df_logs['val_loss'].iloc[-1]) if 'val_loss' in df_logs else np.nan

# Tamanho do arquivo do modelo
model_size_mb = os.path.getsize(model_path) / 1e6

# Complexidade (#parâmetros e FLOPs)
n_params, gflops = get_model_complexity(model, imgsz=IMG_SZ)

# Latência/FPS + recursos/energia (no device definido em LAT_DEVICE)
resources = measure_latency_resources(model, imgsz=IMG_SZ, runs=INFER_RUNS, device=LAT_DEVICE)

# Tempo total de treino
total_train_time_s = read_total_training_time(results_csv)

# ===================== SAÍDAS NO TERMINAL =====================
print("\n===================== RESULTADOS COMPLETOS =====================\n")
print(f"📁 Modelo: {model_path}")
print(f"🔢 Seed={SEED} | K-Fold={KFOLD} | BBox={BBOX} | IMG={IMG_SZ}px | Latência medida em: {LAT_DEVICE}\n")

print("🔹 Métricas gerais (mAP@50, mAP@0.5:0.95, Precision, Recall):")
print(f"  Train:  {fmt(map50_tr)}, {fmt(map5095_tr)}, {fmt(prec_tr)}, {fmt(rec_tr)}")
print(f"  Val:    {fmt(map50_va)}, {fmt(map5095_va)}, {fmt(prec_va)}, {fmt(rec_va)}")
print(f"  Test:   {fmt(map50_te)}, {fmt(map5095_te)}, {fmt(prec_te)}, {fmt(rec_te)}")

print("\n🔹 Métricas por tamanho (val):")
print(f"  mAP@50  (s/m/l): {fmt(size_va.get('map50_s'))} / {fmt(size_va.get('map50_m'))} / {fmt(size_va.get('map50_l'))}")
print(f"  mAP@0.5:0.95 (s/m/l): {fmt(size_va.get('map_s'))} / {fmt(size_va.get('map_m'))} / {fmt(size_va.get('map_l'))}")

print("\n🔹 Recursos e desempenho:")
print(f"  Latência média: {fmt(resources['latency_ms'])} ms  |  FPS: {fmt(resources['fps'])}")
print(f"  RAM média/pico: {fmt(resources['ram'])} / {fmt(resources['ram_peak'])} MB  |  CPU média: {fmt(resources['cpu'])}%")
print(f"  Energia GPU (estimada): {fmt(resources['gpu_energy'])} J (requer NVML)")
print(f"  FLOPs: {fmt(gflops)} GFLOPs  |  Parâmetros: {fmt(n_params)}")
print(f"  Tempo total de treino (s): {fmt(total_train_time_s)}  |  Tamanho do modelo: {fmt(model_size_mb)} MB")

print("\n🔹 Losses finais:")
print(f"  Train Loss: {fmt(loss_train_final)}  |  Val Loss: {fmt(loss_val_final)}")

if cls_names and ap_per_class:
    ap_df = pd.DataFrame({'Classe': cls_names, 'AP@0.5:0.95 (val)': ap_per_class})
    print("\n📊 AP por classe (val):")
    print(ap_df.to_string(index=False))
else:
    print("\n[INFO] AP por classe (val) não disponível nesta versão/execução.")

# Linha Markdown opcional no final (sem salvar, apenas imprime):
print("\n---------------- LINHA MARKDOWN (copiar/colar no README) ----------------")
print(
    f"| **{MODEL_NAME}** | {SEED} | {KFOLD} | {BBOX} | "
    f"{fmt(map50_tr)} | {fmt(map5095_tr)} | {fmt(prec_tr)} | {fmt(rec_tr)} | "
    f"{fmt(map50_va)} | {fmt(map5095_va)} | {fmt(prec_va)} | {fmt(rec_va)} | "
    f"{fmt(map50_te)} | {fmt(map5095_te)} | {fmt(prec_te)} | {fmt(rec_te)} | "
    f"{fmt(resources['latency_ms'])} ms | {fmt(loss_train_final)} | {fmt(loss_val_final)} | {fmt(model_size_mb)} MB |"
)

print("\n✅ Concluído — nenhum arquivo foi salvo; tudo mostrado no terminal.\n")
