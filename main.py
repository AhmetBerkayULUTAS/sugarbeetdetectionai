import cv2
import time
from detector import Detector
from camera import Camera
from metrics import Metrics
from visualizer import Visualizer
# resource_manager.py kaynak yönetimi için ilerde eklenebilir

ENGINE_MODEL_PATH = "model1.engine"
CONF_THRESHOLD = 0.27
NMS_THRESHOLD = 0.30
CLASS_NAMES = ["sugar_beet"]

class LiveDetectionApp:
    def __init__(self, verbose=False):
        self.detector = None
        self._cleaned_up = False
        
        print("TensorRT modeli yükleniyor...")
        try:
            self.detector = Detector(
                ENGINE_MODEL_PATH, 
                conf=CONF_THRESHOLD, 
                iou=NMS_THRESHOLD, 
                verbose=verbose
            )
            print("Model başarıyla yüklendi.")
            
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")

    def run(self):
        if self.detector is None or self._cleaned_up:
            print("Uygulama başlatılamadı. Çıkış yapılıyor.")
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
                if frame is None:
                    continue
                end_acq = time.time()
                metrics.add_acquisition_time((end_acq - start_acq) * 1000)

                # Inference
                start_inf = time.time()
                results = self.detector.infer(frame) 
                end_inf = time.time()
                metrics.add_inference_time((end_inf - start_inf) * 1000)

                # Tespit bilgisini konsola yazdır
                if results:
                    print(f"{len(results)} pancar tespit edildi")
                else:
                    pass

                # Görselleştirme
                elapsed_times = metrics.compute()
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
            self.cleanup(camera)

    def cleanup(self, camera=None):
        """Güvenli cleanup"""
        if self._cleaned_up:
            return
            
        print("Uygulama resource'ları temizleniyor...")
        self._cleaned_up = True
        
        try:
            if camera is not None:
                camera.release()
                print("Kamera temizlendi")
        except Exception as e:
            print(f"Kamera cleanup error: {e}")
        
        try:
            cv2.destroyAllWindows()
            print("OpenCV temizlendi")
        except Exception as e:
            print(f"OpenCV cleanup error: {e}")
        
        try:
            if self.detector is not None:
                self.detector.cleanup()
                print("Detector temizlendi")
        except Exception as e:
            print(f"Detector cleanup error: {e}")
        
        print("Uygulama temizliği tamamlandı")

if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv
    
    app = LiveDetectionApp(verbose=verbose)
    
    try:
        app.run()
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")
    finally:
        app.cleanup()