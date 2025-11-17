from ultralytics import YOLO
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent

    # 👇 Ruta REAL según tu estructura: dataset/placas_colombia/data.yaml
    data_path = base_dir / "dataset" / "placas_colombia" / "data.yaml"

    print("Usando data.yaml en:", data_path)
    print("¿Existe?:", data_path.exists())

    # Si tienes yolov8n.pt en la raíz del proyecto (como en la captura):
    model_path = base_dir / "yolov8n.pt"
    model = YOLO(str(model_path))  # descarga el modelo si no está

    model.train(
        data=str(data_path),
        epochs=50,
        imgsz=640
    )

if __name__ == "__main__":
    main()
