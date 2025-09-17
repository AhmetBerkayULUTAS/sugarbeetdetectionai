from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, conf=0.5, iou=0.45):
        self.model = YOLO(model_path, task='detect', verbose=False)
        self.conf = conf
        self.iou = iou

    def infer(self, frame):
        return self.model(frame, stream=True, conf=self.conf, iou=self.iou, verbose=False)
