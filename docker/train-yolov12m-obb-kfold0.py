from ultralytics import YOLO

# ===== Caminhos =====
DATA_YAML = "/data/dataset.yaml"
MODEL = "yolo12m-obb.yaml"

# ===== Hiperparâmetros =====
EPOCHS = 100
IMGSZ = 1024
BATCH = 4
DEVICE = 0
WORKERS = 0
SEED = 42

# ===== Saída =====
PROJECT = "/app/runs/obb"

def main():
    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        project=PROJECT,
        pretrained=True,
        verbose=True
    )

    print(results)

if __name__ == "__main__":
    main()
