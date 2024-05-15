from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8s.pt')
 
# Training.
results = model.train(
   data='parking_v8.yaml',
   imgsz=640,
   epochs=50,
   batch=-1,
   name='yolov8s_50e_640s_autob')
