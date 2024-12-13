from ultralytics import YOLO

# Build a new model
model = YOLO("yolo11n-cls.yaml")

# Train the model
train_results = model.train(
    data="data",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)