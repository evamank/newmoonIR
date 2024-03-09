from ultralytics import YOLO

# Load a model
if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # build a new model from scratch
    results = model.train(data="config.yaml", epochs=200, imgsz=1000)  # train the model