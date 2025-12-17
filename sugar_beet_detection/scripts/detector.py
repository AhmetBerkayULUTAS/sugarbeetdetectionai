import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit

class Detector:
    def __init__(self, engine_path, conf=0.25, iou=0.45, verbose=False):
        self.conf = conf
        self.iou = iou
        self.engine = None
        self.context = None
        self._cleaned_up = False
        self.verbose = verbose
        
        self.gpu_buffers = []
        self.host_buffers = []
        self.bindings = []
        self.stream = None
        self.letterbox_params = None
        
        # İstatistikler
        self.frame_count = 0
        self.detection_count = 0
        
        if self.verbose:
            print("🔧 TensorRT modeli yükleniyor...")
        
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(engine_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self._allocate_gpu_memory_modern()
            
            if self.verbose:
                print("✅ Model başarıyla yüklendi")
            
        except Exception as e:
            print(f"❌ Model yüklenirken hata: {e}")
            self.cleanup()
            raise

    def _allocate_gpu_memory_modern(self):
        """Modern TensorRT API için memory allocation"""
        try:
            self.stream = cuda.Stream()
            
            # INPUT için (1, 3, 640, 640)
            input_shape = (1, 3, 640, 640)
            input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
            input_gpu = cuda.mem_alloc(input_size)
            input_host = cuda.pagelocked_empty(input_shape, dtype=np.float32)
            
            # DEĞİŞİKLİK: OUTPUT için - (1, 5, 8400) formatına göre
            output_shape = (1, 5, 8400)  # Modelin gerçek output formatı
            output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)
            output_gpu = cuda.mem_alloc(output_size)
            output_host = cuda.pagelocked_empty(output_shape, dtype=np.float32)
            
            self.bindings = [int(input_gpu), int(output_gpu)]
            self.host_buffers = [input_host, output_host]
            self.gpu_buffers = [input_gpu, output_gpu]
            
            # MODERN TENSORRT: Tensor address'leri set et
            if hasattr(self.context, 'set_tensor_address'):
                self.context.set_tensor_address("images", self.bindings[0])
                self.context.set_tensor_address("output0", self.bindings[1])
                if self.verbose:
                    print("✅ Modern TensorRT - Tensor address'ler set edildi")
            
            if self.verbose:
                print(f"✅ Input shape: {input_shape}")
                print(f"✅ Output shape: {output_shape}")  # (1, 5, 8400)
            
        except Exception as e:
            print(f"❌ GPU memory allocation failed: {e}")
            raise

    def infer(self, frame):
        """Ana inference fonksiyonu - TEMİZ ÇIKTI"""
        if frame is None or frame.size == 0:
            return []
            
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        try:
            # Preprocess
            img, letterbox_params = self.preprocess_letterbox(frame)
            self.letterbox_params = letterbox_params
            
            # Sadece verbose mode'da göster
            if self.verbose and self.frame_count % 30 == 0:
                print(f"📐 Frame {self.frame_count}: {w}x{h} -> 640x640")
            
            # GPU inference
            results = self.infer_gpu_optimized(img, h, w)
            
            # Sonuçları göster (her zaman)
            if results:
                self.detection_count += len(results)
                if self.frame_count % 10 == 0:
                    print(f"🌱 Frame {self.frame_count}: {len(results)} pancar - Toplam: {self.detection_count}")
            elif self.frame_count % 50 == 0:
                print(f"🔍 Frame {self.frame_count}: Tespit yok")
                
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Inference error: {e}")
            return []

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Letterbox preprocessing"""
        h, w = img.shape[:2]
        target_h, target_w = new_shape
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        top = pad_h
        bottom = pad_h
        left = pad_w
        right = pad_w
        
        if (target_w - new_w) % 2 != 0:
            right += 1
        if (target_h - new_h) % 2 != 0:
            bottom += 1
        
        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        
        params = {
            'scale': scale,
            'pad_left': left,
            'pad_top': top,
            'original_w': w,
            'original_h': h
        }
        
        return letterboxed, params

    def preprocess_letterbox(self, frame):
        """Preprocessing"""
        letterboxed, params = self.letterbox(frame, new_shape=(640, 640))
        
        img = letterboxed.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img, params

    def infer_gpu_optimized(self, img, orig_h, orig_w):
        """GPU inference - SADECE HATA DURUMUNDA DEBUG"""
        try:
            # Input'u kopyala
            np.copyto(self.host_buffers[0], img)
            cuda.memcpy_htod_async(self.gpu_buffers[0], self.host_buffers[0], self.stream)
            
            # Modern TensorRT için execute
            if hasattr(self.context, 'execute_async_v3'):
                self.context.execute_async_v3(self.stream.handle)
            else:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Output'u al
            cuda.memcpy_dtoh_async(self.host_buffers[1], self.gpu_buffers[1], self.stream)
            self.stream.synchronize()
            
            output_data = self.host_buffers[1]
            
            # SADECE VERBOSE MODE'DA VEYA İLK FRAME'DE GÖSTER
            if self.verbose and self.frame_count == 1:
                print(f"🎯 İlk frame output: {output_data.shape}, range: [{output_data.min():.3f}, {output_data.max():.3f}]")
                non_zero = np.count_nonzero(output_data)
                print(f"🔍 Sıfır olmayan eleman: {non_zero}/{output_data.size}")
            
            return self.post_process_yolov8(output_data, orig_h, orig_w)
            
        except Exception as e:
            print(f"❌ GPU inference error: {e}")
            return []

    def post_process_yolov8(self, output, orig_h, orig_w):
        """
        DEĞİŞİKLİK: (1, 5, 8400) formatı için post-processing
        Format: (1, 5, 8400) where 5 = [x_center, y_center, width, height, confidence]
        """
        try:
            # DEĞİŞİKLİK: Output shape: (1, 5, 8400)
            # Transpose yaparak (8400, 5) formatına getir
            predictions = output[0].transpose(1, 0)  # Shape: (8400, 5)
            
            # Confidence filtering
            mask = predictions[:, 4] >= self.conf
            valid_predictions = predictions[mask]
            
            # SADECE VERBOSE MODE'DA GÖSTER
            if self.verbose and self.frame_count <= 3:
                print(f"🔍 Frame {self.frame_count}: {len(valid_predictions)}/{len(predictions)} prediction")
            
            if len(valid_predictions) == 0:
                # SADECE İLK FRAME'LERDE DEBUG GÖSTER
                if self.verbose and self.frame_count <= 5:
                    print("🔍 İlk frame'lerde tespit yok, model kontrol ediliyor...")
                    # İlk 5 prediction'ı göster (sıfır olsa bile)
                    for i, pred in enumerate(predictions[:5]):
                        print(f"  Prediction {i}: {pred}")
                return []
            
            if self.letterbox_params is None:
                return []
            
            scale = self.letterbox_params['scale']
            pad_left = self.letterbox_params['pad_left']
            pad_top = self.letterbox_params['pad_top']
            
            results = []
            
            for i, pred in enumerate(valid_predictions):
                # DEĞİŞİKLİK: Format: [x_center, y_center, width, height, confidence]
                x_center, y_center, width, height, confidence = pred
                
                # DEĞİŞİKLİK: Center formatından corner formatına çevir
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                # SADECE VERBOSE MODE'DA VE İLK DETECTION'LARI GÖSTER
                if self.verbose and self.frame_count <= 3 and i < 2:
                    print(f"🔍 Frame {self.frame_count} detection {i}: center=({x_center:.1f},{y_center:.1f}), size=({width:.1f},{height:.1f}), conf={confidence:.3f}")
                
                # Letterbox koordinat dönüşümü
                x1_unpadded = (x1 - pad_left)
                y1_unpadded = (y1 - pad_top)
                x2_unpadded = (x2 - pad_left)
                y2_unpadded = (y2 - pad_top)
                
                # Orijinal boyuta çevir
                x1_orig = int(x1_unpadded / scale)
                y1_orig = int(y1_unpadded / scale)
                x2_orig = int(x2_unpadded / scale)
                y2_orig = int(y2_unpadded / scale)
                
                # Boundary check
                x1_orig = max(0, min(x1_orig, orig_w - 1))
                y1_orig = max(0, min(y1_orig, orig_h - 1))
                x2_orig = max(0, min(x2_orig, orig_w - 1))
                y2_orig = max(0, min(y2_orig, orig_h - 1))
                
                bbox_width = x2_orig - x1_orig
                bbox_height = y2_orig - y1_orig
                
                if bbox_width >= 10 and bbox_height >= 10 and x1_orig < x2_orig and y1_orig < y2_orig:
                    results.append({
                        "box": [x1_orig, y1_orig, x2_orig, y2_orig],
                        "score": float(confidence),
                        "class_id": 0  # Tek sınıf için
                    })
            
            # NMS
            if len(results) > 1 and self.iou > 0:
                results = self._apply_nms(results)
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Post-processing error: {e}")
            return []

    def _apply_nms(self, detections):
        """Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
            
        try:
            boxes = np.array([d["box"] for d in detections])
            scores = np.array([d["score"] for d in detections])
            
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]
            
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                
                inds = np.where(ovr <= self.iou)[0]
                order = order[inds + 1]
            
            return [detections[i] for i in keep]
            
        except Exception as e:
            return detections

    def cleanup(self):
        """Cleanup"""
        if self._cleaned_up:
            return
        
        if self.verbose:
            print("🧹 Temizlik yapılıyor...")
        
        self._cleaned_up = True
        
        try:
            # Stream sync
            if self.stream:
                try:
                    self.stream.synchronize()
                except:
                    pass
            
            # GPU memory
            for gpu_buffer in self.gpu_buffers:
                if gpu_buffer:
                    try:
                        gpu_buffer.free()
                    except:
                        pass
            
            self.gpu_buffers.clear()
            self.host_buffers.clear()
            self.bindings.clear()
            
            # Context ve engine
            if self.context:
                try:
                    del self.context
                    self.context = None
                except:
                    pass
            
            if self.engine:
                try:
                    del self.engine
                    self.engine = None
                except:
                    pass
            
            if self.stream:
                try:
                    self.stream = None
                except:
                    pass
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Cleanup error: {e}")
        
        if self.verbose:
            print(f"📊 Özet: {self.frame_count} frame, {self.detection_count} tespit")
            print("✅ Cleanup tamamlandı")

    def __del__(self):
        if not self._cleaned_up:
            self.cleanup()