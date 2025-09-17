import cv2
import time
from detector import Detector
from camera import Camera
from metrics import Metrics
from visualizer import Visualizer

ENGINE_MODEL_PATH = "model1.engine"
CONF_THRESHOLD = 0.270 #model1=2.7, model2=0.299
NMS_THRESHOLD = 0.50
CLASS_NAMES = ["sugar_beet"]

class LiveDetectionApp:
    def __init__(self):
        print("TensorRT modeli yükleniyor...")
        try:
            self.detector = Detector(ENGINE_MODEL_PATH, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD)
            print("Model yüklendi.")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")

    def run(self):
        if self.detector is None:
            print("Uygulama başlatılamadı çünkü model yüklenemedi. Çıkış yapılıyor.")
            return
        
        camera = Camera()
        metrics = Metrics()
        visualizer = Visualizer(CLASS_NAMES)

        print("Kamera akışı başladı. Çıkmak için 'q' tuşuna basın.")

        try:
            while True:
                # Frame al
                start_acq = time.time()
                frame = camera.get_frame()
                end_acq = time.time()
                metrics.add_acquisition_time((end_acq - start_acq) * 1000)

                # İnference 
                start_inf = time.time()
                results = self.detector.infer(frame) 
                end_inf = time.time()
                metrics.add_inference_time((end_inf - start_inf) * 1000)

                # Metrikleri hesapla
                elapsed_times = metrics.compute()

                # Görselleştirme
                annotated = visualizer.draw(frame, results, elapsed_times)
                if annotated is not None:
                    cv2.imshow("Pancar Algilama (TensorRT)", annotated)

                if cv2.waitKey(1) == ord('q'):
                    break
        except KeyboardInterrupt:
            print("\nKullanıcı tarafından durduruldu.")
        except Exception as e:
            print(f"Bir hata oluştu: {e}")
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = LiveDetectionApp()  
    app.run()
    