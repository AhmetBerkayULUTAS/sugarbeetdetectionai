import cv2

class Camera:
    def __init__(self, cam_id=0, width=640, height=640):
        
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError("Kamera açılamadı!")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Boş kare okundu!")
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

#jetson nanoda csi kamera için gstream eklenebilir
#Kamera ayarlarınızı mümkün olan en yüksek çözünürlüğe getirip, ardından modeli çalıştırmadan önce görüntüyü küçültmeye gidilebilir