"""
Train YOLOv8 on "Fire Smoke and Human Detector" dataset from Roboflow.

Dataset: https://universe.roboflow.com/spyrobot/fire-smoke-and-human-detector
Classes: fire, human, smoke (3 classes)
Format: YOLOv8

Instructions:
1. Download the dataset from Roboflow in YOLOv8 format
2. Place the downloaded folder inside images/data/
3. Run this script to train YOLOv8n for 10 epochs
"""
import sys
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.stdout.reconfigure(encoding='utf-8')

    from ultralytics import YOLO

    DATA_YAML = r"D:\USA\Sems\4_Spring 2026 Sem\AI for Engineers\Multimodal\lokesh\images\images\data\data.yaml"
    PROJECT_DIR = r"D:\USA\Sems\4_Spring 2026 Sem\AI for Engineers\Multimodal\lokesh\images\images"

    print("=" * 60)
    print("Training YOLOv8n on Fire/Smoke/Human Detector Dataset")
    print("Source: Roboflow 'Fire Smoke and Human Detector'")
    print("Classes: fire, human, smoke")
    print("=" * 60)

    print("\nLoading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    print("Starting training (10 epochs, 640px, batch 16)...")
    results = model.train(
        data=DATA_YAML,
        epochs=10,
        imgsz=640,
        batch=16,
        name="fire_detector",
        project=PROJECT_DIR,
        patience=5,
        workers=0,
        device=0,  # GPU (use 'cpu' if no GPU)
        exist_ok=True,
        verbose=True,
    )
    print("\nTRAINING COMPLETE!")
    print(f"Best model: {PROJECT_DIR}/fire_detector/weights/best.pt")
