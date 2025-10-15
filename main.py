import cv2
import time
from detector import Detector
from camera import Camera
from metrics import Metrics
from visualizer import Visualizer

ENGINE_MODEL_PATH = "best.engine"
CONF_THRESHOLD = 0.5 #model1:0.27 model2:0.52
NMS_THRESHOLD = 0.30
CLASS_NAMES = ["sugar_beet"]

class LiveDetectionApp:
    def __init__(self, camera_id=0, verbose=False):
        """
        Canlı tespit uygulaması
        
        Args:
            camera_id: USB kamera ID (0, 1, 2...)
            verbose: Detaylı log
        """
        self.detector = None
        self.camera = None
        self._cleaned_up = False
        self.verbose = verbose
        self.camera_id = camera_id
        
        print("PANCAR TESPİT SİSTEMİ")
        
        # Model yükle
        print("\n📦 TensorRT modeli yükleniyor...")
        try:
            self.detector = Detector(
                ENGINE_MODEL_PATH, 
                conf=CONF_THRESHOLD, 
                iou=NMS_THRESHOLD, 
                verbose=verbose
            )
            print("✅ Model başarıyla yüklendi")
            
        except Exception as e:
            print(f"❌ Model yüklenirken hata oluştu: {e}")
            raise

    def initialize_camera(self):
        """Kamerayı başlat ve boyutları öğren"""
        print("\n📷 Kamera başlatılıyor...")
        
        try:
            # USB/Webcam
            self.camera = Camera(
                cam_id=self.camera_id,
                preferred_width=None,  # Kameranın varsayılanı
                preferred_height=None,
                verbose=self.verbose
            )
            
            # Kamera çözünürlüğünü al
            cam_width, cam_height = self.camera.get_resolution()
            print(f"✅ Kamera hazır: {cam_width}x{cam_height}")
            
            # Desteklenen çözünürlükleri listele (isteğe bağlı)
            if self.verbose:
                print("\n🔍 Desteklenen çözünürlükler kontrol ediliyor...")
                supported = self.camera.list_supported_resolutions()
                print(f"✅ Toplam {len(supported)} çözünürlük destekleniyor:")
                for w, h in supported:
                    print(f"   - {w}x{h}")
            
            return True
            
        except Exception as e:
            print(f"❌ Kamera başlatılamadı: {e}")
            return False

    def run(self):
        """Ana döngü"""
        if self.detector is None or self._cleaned_up:
            print("❌ Uygulama başlatılamadı. Çıkış yapılıyor.")
            return
        
        # Kamerayı başlat
        if not self.initialize_camera():
            return
        
        metrics = Metrics()
        visualizer = Visualizer(CLASS_NAMES)

        print("\n" + "=" * 60)
        print("🎬 CANLI GÖRÜNTÜ BAŞLADI")
        print("   Çıkmak için 'q' tuşuna basın")
        print("   Ekran görüntüsü için 's' tuşuna basın")
        print("=" * 60 + "\n")

        frame_count = 0
        screenshot_count = 0

        try:
            while True:
                # Frame al
                start_acq = time.time()
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                end_acq = time.time()
                metrics.add_acquisition_time((end_acq - start_acq) * 1000)

                # Frame sayısı
                frame_count += 1

                # Inference
                start_inf = time.time()
                results = self.detector.infer(frame) 
                end_inf = time.time()
                metrics.add_inference_time((end_inf - start_inf) * 1000)

                # Tespit bilgisini konsola yazdır
                if results:
                    if self.verbose or frame_count % 30 == 0:  # Her 30 frame'de bir veya verbose mode
                        print(f"🌱 Frame {frame_count}: {len(results)} pancar tespit edildi")

                # Görselleştirme
                elapsed_times = metrics.compute()
                annotated = visualizer.draw(frame, results, elapsed_times)
                
                if annotated is not None:
                    cv2.imshow("Pancar Algılama (TensorRT)", annotated)

                # Klavye kontrolleri
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n⏹️  Kullanıcı tarafından durduruldu")
                    break
                elif key == ord('s'):
                    # Ekran görüntüsü kaydet
                    screenshot_count += 1
                    filename = f"screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"📸 Ekran görüntüsü kaydedildi: {filename}")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Keyboard interrupt (Ctrl+C)")
        except Exception as e:
            print(f"\n❌ Bir hata oluştu: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Güvenli cleanup"""
        if self._cleaned_up:
            return
            
        print("\n🧹 Kaynaklar temizleniyor...")
        self._cleaned_up = True
        
        try:
            if self.camera is not None:
                self.camera.release()
                print("  ✅ Kamera temizlendi")
        except Exception as e:
            print(f"  ⚠️  Kamera cleanup error: {e}")
        
        try:
            cv2.destroyAllWindows()
            print("  ✅ OpenCV temizlendi")
        except Exception as e:
            print(f"  ⚠️  OpenCV cleanup error: {e}")
        
        try:
            if self.detector is not None:
                self.detector.cleanup()
                print("  ✅ Detector temizlendi")
        except Exception as e:
            print(f"  ⚠️  Detector cleanup error: {e}")
        
        print("✅ Temizlik tamamlandı")


def print_help():
    """Yardım mesajı"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           PANCAR TESPİT SİSTEMİ - KULLANIM                  ║
╚══════════════════════════════════════════════════════════════╝

Kullanım:
  python main.py [seçenekler]

Seçenekler:
  --verbose          Detaylı log göster
  --camera-id N      Kamera ID (varsayılan: 0)
  --help             Bu yardım mesajını göster

Örnekler:
  python main.py                    # USB kamera (ID=0)
  python main.py --verbose          # Detaylı log
  python main.py --camera-id 1      # USB kamera (ID=1)

Klavye Kısayolları:
  q - Çıkış
  s - Ekran görüntüsü kaydet
    """)


if __name__ == "__main__":
    import sys
    
    # Komut satırı argümanları
    verbose = "--verbose" in sys.argv
    show_help = "--help" in sys.argv or "-h" in sys.argv
    
    # Kamera ID
    camera_id = 0
    if "--camera-id" in sys.argv:
        try:
            idx = sys.argv.index("--camera-id")
            camera_id = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("❌ Geçersiz camera-id değeri!")
            sys.exit(1)
    
    # Yardım göster
    if show_help:
        print_help()
        sys.exit(0)
    
    # Uygulamayı başlat
    app = LiveDetectionApp(
        camera_id=camera_id,
        verbose=verbose
    )
    
    try:
        app.run()
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.cleanup()