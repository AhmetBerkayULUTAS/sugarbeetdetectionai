#!/usr/bin/env python3
"""
ai_publisher_node.py - JETSON NANO'DA ÇALIŞIR
Kameradan görüntü alır, AI ile işler ve ROS2 topic'lerine yayınlar
Host bilgisayardaki UI Node ile iletişim kurar
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Float32, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import sys
import os

# Mevcut dosyalarınızı import et
from detector import Detector
from camera import Camera, CSICamera

# Custom message'ları import et (build sonrası çalışacak)
try:
    from sugar_beet_detection.msg import DetectionData, SystemMetrics
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    print("⚠️  Custom messages bulunamadı. Build yapıldığından emin olun.")
    CUSTOM_MSGS_AVAILABLE = False


class AIPubNode(Node):
    def __init__(self):
        super().__init__('ai_publisher_node')
        
        # Parametreler - launch file'dan gelecek
        self.declare_parameter('camera_type', 'usb')
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('engine_path', 'model2.engine')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.30)
        self.declare_parameter('publish_rate', 30)
        self.declare_parameter('send_annotated', True)
        self.declare_parameter('verbose', False)
        
        # Parametreleri al
        self.camera_type = self.get_parameter('camera_type').value
        self.camera_id = self.get_parameter('camera_id').value
        self.engine_path = self.get_parameter('engine_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.send_annotated = self.get_parameter('send_annotated').value
        self.verbose = self.get_parameter('verbose').value
        
        # Başlangıç mesajı
        self.get_logger().info("=" * 70)
        self.get_logger().info("AI PUBLISHER NODE - JETSON NANO")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"📷 Camera: {self.camera_type} (ID: {self.camera_id})")
        self.get_logger().info(f"🤖 Model: {self.engine_path}")
        self.get_logger().info(f"🎯 Confidence: {self.conf_threshold}")
        self.get_logger().info(f"📊 Publish rate: {self.publish_rate} Hz")
        self.get_logger().info(f"🎨 Annotated: {self.send_annotated}")
        self.get_logger().info("=" * 70)
        
        # CvBridge - OpenCV <-> ROS2 dönüşümü
        self.bridge = CvBridge()
        
        # Publishers - UI Node'unuzun beklediği topic'ler
        self.get_logger().info("\n📡 Publishers oluşturuluyor...")
        
        # 1. Görüntü publisher
        self.image_pub = self.create_publisher(Image, 'ai_output_image', 10)
        self.get_logger().info("  ✅ /ai_output_image")
        
        # 2. Detections publisher
        if CUSTOM_MSGS_AVAILABLE:
            self.detections_pub = self.create_publisher(DetectionData, 'ai_detections', 10)
            self.get_logger().info("  ✅ /ai_detections (DetectionData)")
        else:
            self.get_logger().warn("  ⚠️  DetectionData mesajı yok, build yapın!")
        
        # 3. Inference time publisher
        self.inference_time_pub = self.create_publisher(Int32, 'ai_inference_time', 10)
        self.get_logger().info("  ✅ /ai_inference_time")
        
        # 4. FPS publisher
        self.fps_pub = self.create_publisher(Int32, 'fps', 10)
        self.get_logger().info("  ✅ /fps")
        
        # 5. System metrics publisher
        if CUSTOM_MSGS_AVAILABLE:
            self.metrics_pub = self.create_publisher(SystemMetrics, 'system_metrics', 10)
            self.get_logger().info("  ✅ /system_metrics (SystemMetrics)")
        
        # 6. Camera check publishers (6 kamera için)
        self.camera_check_pubs = {}
        for i in range(1, 7):
            self.camera_check_pubs[i] = self.create_publisher(
                Bool, f'camera{i}_check', 10
            )
        self.get_logger().info("  ✅ /camera1_check ... /camera6_check")
        
        # 7. Image acquisition time
        self.acq_time_pub = self.create_publisher(Int32, 'image_acquisition_time', 10)
        self.get_logger().info("  ✅ /image_acquisition_time")
        
        # 8. Image analysis time (inference time ile aynı)
        self.analysis_time_pub = self.create_publisher(Int32, 'image_analysis_time', 10)
        self.get_logger().info("  ✅ /image_analysis_time")
        
        # Detector ve Camera
        self.detector = None
        self.camera = None
        
        # İstatistikler
        self.frame_count = 0
        self.total_detections = 0
        self.start_time = time.time()
        
        # FPS hesaplama
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Metrik ortalamaları (son 30 frame)
        self.acquisition_times = []
        self.inference_times = []
        
        # Sistem başlat
        if not self.initialize():
            self.get_logger().error("❌ Sistem başlatılamadı!")
            raise RuntimeError("Initialization failed")
        
        # Timer - ana döngü
        timer_period = 1.0 / self.publish_rate  # saniye
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info("\n✅ Node hazır - Yayın başlıyor...\n")
    
    def initialize(self):
        """Kamera ve AI modelini başlat"""
        try:
            self.get_logger().info("\n🔧 SİSTEM BAŞLATILIYOR...")
            
            # 1. Kamera başlat
            self.get_logger().info(f"\n📷 {self.camera_type.upper()} kamera başlatılıyor...")
            
            if self.camera_type.lower() == "csi":
                # Jetson Nano CSI kamera
                self.camera = CSICamera(
                    sensor_id=self.camera_id,
                    width=1280,
                    height=720,
                    fps=30,
                    flip_method=0,
                    verbose=self.verbose
                )
            else:
                # USB kamera
                self.camera = Camera(
                    cam_id=self.camera_id,
                    preferred_width=640,
                    preferred_height=480,
                    verbose=self.verbose
                )
            
            cam_width, cam_height = self.camera.get_resolution()
            self.get_logger().info(f"✅ Kamera hazır: {cam_width}x{cam_height}")
            
            # Kamera durumunu yayınla (kamera 1 aktif)
            self.camera_check_pubs[1].publish(Bool(data=True))
            
            # 2. AI Model yükle
            self.get_logger().info(f"\n🤖 TensorRT modeli yükleniyor...")
            self.get_logger().info(f"   Model yolu: {self.engine_path}")
            
            if not os.path.exists(self.engine_path):
                self.get_logger().error(f"❌ Model dosyası bulunamadı: {self.engine_path}")
                return False
            
            self.detector = Detector(
                self.engine_path,
                conf=self.conf_threshold,
                iou=self.nms_threshold,
                verbose=self.verbose
            )
            self.get_logger().info("✅ AI model yüklendi")
            
            self.get_logger().info("\n" + "=" * 70)
            self.get_logger().info("✅ SİSTEM HAZIR")
            self.get_logger().info("=" * 70)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Başlatma hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def timer_callback(self):
        """Ana döngü - her frame'de çalışır"""
        try:
            # 1. Frame al (acquisition time ölç)
            acq_start = time.time()
            frame = self.camera.get_frame()
            acq_time = (time.time() - acq_start) * 1000  # ms
            
            if frame is None:
                self.get_logger().warn("⚠️  Boş frame alındı")
                return
            
            self.frame_count += 1
            
            # 2. AI inference (inference time ölç)
            inf_start = time.time()
            results = self.detector.infer(frame)
            inf_time = (time.time() - inf_start) * 1000  # ms
            
            # 3. Metrikleri kaydet (son 30 frame için ortalama)
            self.acquisition_times.append(acq_time)
            self.inference_times.append(inf_time)
            if len(self.acquisition_times) > 30:
                self.acquisition_times.pop(0)
            if len(self.inference_times) > 30:
                self.inference_times.pop(0)
            
            # 4. Tespit sayısı
            num_detections = len(results)
            if results:
                self.total_detections += num_detections
            
            # 5. DetectionData mesajı oluştur ve yayınla
            if CUSTOM_MSGS_AVAILABLE and hasattr(self, 'detections_pub'):
                detection_msg = self.create_detection_message(results)
                self.detections_pub.publish(detection_msg)
            
            # 6. Görselleştirilmiş görüntü oluştur
            if self.send_annotated and results:
                annotated = self.draw_detections(frame.copy(), results)
            else:
                annotated = frame
            
            # 7. Image mesajı yayınla
            try:
                img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = "camera_frame"
                self.image_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"❌ Image publish error: {e}")
            
            # 8. Inference time yayınla
            self.inference_time_pub.publish(Int32(data=int(inf_time)))
            self.analysis_time_pub.publish(Int32(data=int(inf_time)))
            
            # 9. Acquisition time yayınla
            self.acq_time_pub.publish(Int32(data=int(acq_time)))
            
            # 10. FPS hesapla ve yayınla
            self.update_fps()
            self.fps_pub.publish(Int32(data=int(self.current_fps)))
            
            # 11. Sistem metrikleri yayınla
            if CUSTOM_MSGS_AVAILABLE and hasattr(self, 'metrics_pub'):
                metrics_msg = self.create_metrics_message(
                    acq_time, inf_time, num_detections, results
                )
                self.metrics_pub.publish(metrics_msg)
            
            # 12. Log (her 30 frame'de)
            if self.frame_count % 30 == 0:
                avg_conf = np.mean([r['score'] for r in results]) if results else 0.0
                self.get_logger().info(
                    f"📊 Frame {self.frame_count:5d} | "
                    f"FPS: {self.current_fps:5.1f} | "
                    f"Det: {num_detections:2d} | "
                    f"Inf: {inf_time:5.1f}ms | "
                    f"Acq: {acq_time:4.1f}ms | "
                    f"Conf: {avg_conf:.2f}"
                )
            
        except Exception as e:
            self.get_logger().error(f"❌ Timer callback error: {e}")
            import traceback
            traceback.print_exc()
    
    def create_detection_message(self, results):
        """DetectionData mesajı oluştur"""
        msg = DetectionData()
        msg.count = len(results)
        
        if results:
            msg.confidences = [float(r['score']) for r in results]
            msg.x1 = [int(r['box'][0]) for r in results]
            msg.y1 = [int(r['box'][1]) for r in results]
            msg.x2 = [int(r['box'][2]) for r in results]
            msg.y2 = [int(r['box'][3]) for r in results]
            msg.class_ids = [int(r['class_id']) for r in results]
        
        return msg
    
    def create_metrics_message(self, acq_time, inf_time, num_det, results):
        """SystemMetrics mesajı oluştur"""
        msg = SystemMetrics()
        msg.fps = self.current_fps
        msg.inference_time_ms = inf_time
        msg.acquisition_time_ms = acq_time
        msg.total_latency_ms = acq_time + inf_time
        msg.frame_count = self.frame_count
        msg.total_detections = self.total_detections
        
        if results:
            msg.avg_confidence = float(np.mean([r['score'] for r in results]))
        else:
            msg.avg_confidence = 0.0
        
        return msg
    
    def draw_detections(self, frame, results):
        """Tespit sonuçlarını frame üzerine çiz"""
        color = (0, 255, 0)  # Yeşil
        
        for det in results:
            x1, y1, x2, y2 = det['box']
            score = det['score']
            
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"pancar: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stats overlay
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Det: {len(results)}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def update_fps(self):
        """FPS hesapla (son 30 frame için)"""
        self.fps_frame_count += 1
        if self.fps_frame_count >= 30:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_frame_count / elapsed if elapsed > 0 else 0
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
    
    def destroy_node(self):
        """Cleanup - node kapatılırken"""
        self.get_logger().info("\n🧹 Kaynaklar temizleniyor...")
        
        # Kamera kapat
        if self.camera is not None:
            self.camera.release()
            self.get_logger().info("  ✅ Kamera temizlendi")
        
        # Detector cleanup
        if self.detector is not None:
            self.detector.cleanup()
            self.get_logger().info("  ✅ Detector temizlendi")
        
        # İstatistikler
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        self.get_logger().info(f"\n📊 ÖZET:")
        self.get_logger().info(f"  Toplam frame: {self.frame_count}")
        self.get_logger().info(f"  Toplam tespit: {self.total_detections}")
        self.get_logger().info(f"  Ortalama FPS: {avg_fps:.2f}")
        self.get_logger().info(f"  Çalışma süresi: {elapsed:.1f}s")
        
        if self.acquisition_times:
            avg_acq = np.mean(self.acquisition_times)
            self.get_logger().info(f"  Ortalama acquisition: {avg_acq:.1f}ms")
        
        if self.inference_times:
            avg_inf = np.mean(self.inference_times)
            self.get_logger().info(f"  Ortalama inference: {avg_inf:.1f}ms")
        
        self.get_logger().info("✅ Temizlik tamamlandı")
        
        super().destroy_node()


def main(args=None):
    """Ana fonksiyon"""
    rclpy.init(args=args)
    
    try:
        node = AIPubNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n⏹️  Keyboard interrupt (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()