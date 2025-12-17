import cv2
import logging

class Camera:
    def __init__(self, cam_id=0, preferred_width=None, preferred_height=None, verbose=False):
        """
        USB/Webcam sınıfı - Otomatik boyut algılama
        
        Args:
            cam_id: Kamera ID (0, 1, 2...)
            preferred_width: Tercih edilen genişlik (None ise kameranın varsayılanı)
            preferred_height: Tercih edilen yükseklik (None ise kameranın varsayılanı)
            verbose: Detaylı log göster
        """
        self.logger = logging.getLogger("Camera")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.cap = cv2.VideoCapture(cam_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Kamera açılamadı! (cam_id={cam_id})")
        
        # Eğer tercih edilen boyutlar verilmişse ayarla
        if preferred_width and preferred_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)
            self.logger.info(f"🎯 Tercih edilen çözünürlük ayarlandı: {preferred_width}x{preferred_height}")
        
        # Gerçek kamera boyutlarını oku
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Backend bilgisi
        backend = self.cap.getBackendName()
        
        self.logger.info("📷 KAMERA BİLGİLERİ")
        self.logger.info(f"  Kamera ID: {cam_id}")
        self.logger.info(f"  Backend: {backend}")
        self.logger.info(f"  Çözünürlük: {self.width}x{self.height}")
        self.logger.info(f"  FPS: {self.fps}")
        
        # İlk frame'i test et
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            raise RuntimeError("❌ Kameradan frame okunamadı!")
        
        actual_h, actual_w = test_frame.shape[:2]
        
        # Eğer okunan frame boyutu farklıysa güncelle
        if actual_w != self.width or actual_h != self.height:
            self.logger.warning(f"⚠️  Frame boyutu farklı: {actual_w}x{actual_h} (beklenen: {self.width}x{self.height})")
            self.width = actual_w
            self.height = actual_h
            self.logger.info(f"✅ Güncellendi: {self.width}x{self.height}")

    def get_frame(self):
        """
        Kameradan frame oku
        
        Returns:
            frame: BGR formatında görüntü (H, W, 3)
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("❌ Boş kare okundu!")
        return frame

    def get_resolution(self):
        """
        Kameranın gerçek çözünürlüğünü döndür
        
        Returns:
            (width, height): Kamera boyutları
        """
        return self.width, self.height

    def set_resolution(self, width, height):
        """
        Kamera çözünürlüğünü değiştir
        
        Args:
            width: Yeni genişlik
            height: Yeni yükseklik
            
        Returns:
            success: Başarılı olup olmadığı
        """
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Gerçek değerleri oku
        new_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        new_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if new_width == width and new_height == height:
            self.width = new_width
            self.height = new_height
            self.logger.info(f"✅ Çözünürlük değiştirildi: {width}x{height}")
            return True
        else:
            self.logger.warning(f"⚠️  İstenen çözünürlük ayarlanamadı: {width}x{height}")
            self.logger.warning(f"   Gerçek çözünürlük: {new_width}x{new_height}")
            self.width = new_width
            self.height = new_height
            return False

    def list_supported_resolutions(self):
        """
        Yaygın çözünürlükleri test et ve desteklenenleri listele
        
        Returns:
            supported: Desteklenen çözünürlük listesi
        """
        common_resolutions = [
            (320, 240),    # QVGA
            (640, 480),    # VGA
            (800, 600),    # SVGA
            (1024, 768),   # XGA
            (1280, 720),   # HD
            (1280, 960),   # SXGA-
            (1920, 1080),  # Full HD
            (2560, 1440),  # 2K
        ]
        
        supported = []
        original_w, original_h = self.width, self.height
        
        self.logger.info("🔍 Desteklenen çözünürlükler test ediliyor...")
        
        for w, h in common_resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_w == w and actual_h == h:
                supported.append((w, h))
                self.logger.debug(f"  ✅ {w}x{h}")
            else:
                self.logger.debug(f"  ❌ {w}x{h} → {actual_w}x{actual_h}")
        
        # Orijinal çözünürlüğe geri dön
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_h)
        self.width = original_w
        self.height = original_h
        
        self.logger.info(f"✅ {len(supported)} çözünürlük destekleniyor")
        return supported

    def release(self):
        """Kamerayı kapat"""
        if self.cap.isOpened():
            self.cap.release()
            self.logger.info("📷 Kamera kapatıldı")

    def __del__(self):
        """Destructor - Otomatik temizlik"""
        self.release()


# Jetson Nano için CSI kamera desteği
class CSICamera(Camera):
    """
    Jetson Nano CSI kamera için GStreamer pipeline
    """
    def __init__(self, sensor_id=0, width=1280, height=720, fps=30, flip_method=0, verbose=False):
        """
        CSI Kamera (Jetson Nano)
        
        Args:
            sensor_id: CSI kamera ID (0 veya 1)
            width: Genişlik
            height: Yükseklik
            fps: Frame per second
            flip_method: Görüntü döndürme (0-6 arası)
        """
        self.logger = logging.getLogger("CSICamera")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # GStreamer pipeline
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
            f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink"
        )
        
        self.logger.info("🎥 CSI Kamera (GStreamer) başlatılıyor...")
        self.logger.debug(f"Pipeline: {gst_pipeline}")
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            raise RuntimeError("❌ CSI kamera açılamadı!")
        
        self.width = width
        self.height = height
        self.fps = fps
        
        self.logger.info("=" * 50)
        self.logger.info("📷 CSI KAMERA BİLGİLERİ")
        self.logger.info("=" * 50)
        self.logger.info(f"  Sensor ID: {sensor_id}")
        self.logger.info(f"  Çözünürlük: {self.width}x{self.height}")
        self.logger.info(f"  FPS: {self.fps}")
        self.logger.info(f"  Flip Method: {flip_method}")
        self.logger.info("=" * 50)