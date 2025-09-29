import tensorrt as trt
import numpy as np
import cv2
import logging
import pycuda.driver as cuda
import pycuda.autoinit

class Detector:
    def __init__(self, engine_path, conf=0.25, iou=0.45, verbose=False):
        self.conf = conf
        self.iou = iou
        self.engine = None
        self.context = None
        self._cleaned_up = False
        
        # GPU MEMORY MANAGEMENT
        self.gpu_buffers = []
        self.host_buffers = []
        self.bindings = []
        self.stream = None
        
        # Logging setup
        self.logger = logging.getLogger("Detector")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)
        
        try:
            self.logger.debug("TensorRT modeli yükleniyor...")
            
            # TensorRT runtime
            runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            with open(engine_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            # Tensor isimlerini ve şekillerini al
            self.input_name = "images"
            self.output_name = "output0"
            
            # Input shape'i set et
            self.context.set_input_shape(self.input_name, (1, 3, 640, 640))
            
            # Bellek ayırma
            self._allocate_gpu_memory()
            
            self.logger.info("Model başarıyla yüklendi")
            self.logger.info(f"Input: {self.input_name}, shape: (1, 3, 640, 640)")
            self.logger.info(f"Output: {self.output_name}, shape: (1, 300, 6)")
            
        except Exception as e:
            self.logger.error(f"Model yüklenirken hata oluştu: {e}")
            self.cleanup()
            raise

    def _allocate_gpu_memory(self):
        """GPU memory allocation - DÜZELTİLMİŞ VERSİYON"""
        try:
            self.stream = cuda.Stream()
            
            # Input için bellek ayır (1, 3, 640, 640)
            input_shape = (1, 3, 640, 640)
            
            # int64'ü int'e çevir
            input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
            input_gpu = cuda.mem_alloc(input_size)
            input_host = cuda.pagelocked_empty(input_shape, dtype=np.float32)
            
            # Output için bellek ayır (1, 300, 6)
            output_shape = (1, 300, 6)
            
            # int64'ü int'e çevir
            output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)
            output_gpu = cuda.mem_alloc(output_size)
            output_host = cuda.pagelocked_empty(output_shape, dtype=np.float32)
            
            # Listelere ekle
            self.gpu_buffers = [input_gpu, output_gpu]
            self.host_buffers = [input_host, output_host]
            self.bindings = [int(input_gpu), int(output_gpu)]
            
            self.logger.info(f"GPU memory allocated: input={input_size}, output={output_size} bytes")
            
        except Exception as e:
            self.logger.error(f"GPU memory allocation failed: {e}")
            raise

    def infer(self, frame):
        if frame is None or frame.size == 0 or self._cleaned_up:
            return []
            
        h, w = frame.shape[:2]
        
        try:
            # Preprocess
            img = self.preprocess(frame)
            
            # GPU inference
            results = self.infer_gpu_optimized(img, h, w)
            return results
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return []

    def infer_gpu_optimized(self, img, orig_h, orig_w):
        """GPU-optimized inference"""
        try:
            # 1. Input'u host buffer'a kopyala
            np.copyto(self.host_buffers[0], img)
            
            # 2. Host → Device async copy
            cuda.memcpy_htod_async(self.gpu_buffers[0], self.host_buffers[0], self.stream)
            
            # 3. Tensor adreslerini set et
            self.context.set_tensor_address(self.input_name, self.bindings[0])
            self.context.set_tensor_address(self.output_name, self.bindings[1])
            
            # 4. Inference çalıştır
            self.context.execute_async_v3(self.stream.handle)
            
            # 5. Output'u Device → Host kopyala
            cuda.memcpy_dtoh_async(self.host_buffers[1], self.gpu_buffers[1], self.stream)
            
            # 6. Stream'i bekle
            self.stream.synchronize()
            
            # 7. Post-processing
            return self.post_process(self.host_buffers[1], orig_h, orig_w)
            
        except Exception as e:
            self.logger.error(f"GPU inference error: {e}")
            return []

    def preprocess(self, frame):
        """Preprocessing - optimized"""
        h_in, w_in = 640, 640
        
        # Resize
        img = cv2.resize(frame, (w_in, h_in))
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img

    def post_process(self, output, orig_h, orig_w):
        """OPTIMIZED POST-PROCESSING: Vectorized operations ile"""
        try:
            detections = output[0]  # (300, 6)
            
            # OPTIMIZATION 1: Vectorized confidence filtering
            mask = detections[:, 4] >= self.conf
            valid_detections = detections[mask]
            
            if len(valid_detections) == 0:
                return []
            
            results = []
            
            # OPTIMIZATION 2: Sadece geçerli detection'ları işle
            for det in valid_detections:
                x1, y1, x2, y2, confidence, class_id = det
                
                # Bbox'ları orijinal boyuta scale et
                x1 = int(x1 * orig_w / 640)
                y1 = int(y1 * orig_h / 640)
                x2 = int(x2 * orig_w / 640)
                y2 = int(y2 * orig_h / 640)
                
                # Boundary check
                x1 = max(0, min(x1, orig_w-1))
                y1 = max(0, min(y1, orig_h-1))
                x2 = max(0, min(x2, orig_w-1))
                y2 = max(0, min(y2, orig_h-1))
                
                # Geçerli bbox kontrolü
                if x1 < x2 and y1 < y2:
                    results.append({
                        "box": [x1, y1, x2, y2],
                        "score": float(confidence),
                        "class_id": int(class_id)
                    })
            
            # OPTIMIZATION 3: Sadece detection varsa log yaz
            if results:
                self.logger.debug(f"{len(results)} detection bulundu (NMS öncesi)")
            
            # NMS uygula
            if len(results) > 1 and self.iou > 0:
                results = self._apply_nms(results)
                
            if results:
                self.logger.debug(f"{len(results)} detection (NMS sonrası)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Post-processing error: {e}")
            return []

    def _apply_nms(self, detections):
        """Optimized Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
            
        try:
            boxes = np.array([d["box"] for d in detections])
            scores = np.array([d["score"] for d in detections])
            
            # NMS implementation
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
            self.logger.warning(f"NMS uygulanamadı: {e}")
            return detections   

    def cleanup(self):
        """Safe cleanup"""
        if self._cleaned_up:
            return
        
        self.logger.info("TensorRT ve GPU resource'ları temizleniyor...")
        self._cleaned_up = True
        
        # GPU memory temizle
        try:
            for gpu_buffer in self.gpu_buffers:
                if gpu_buffer:
                    gpu_buffer.free()
            
            self.gpu_buffers.clear()
            self.host_buffers.clear()
            self.bindings.clear()
            
            if self.stream:
                self.stream = None
                
            self.logger.debug("GPU memory freed")
            
        except Exception as e:
            self.logger.warning(f"GPU memory cleanup error: {e}")
        
        # Diğer temizlikler
        try:
            if self.context:
                del self.context
                self.context = None
        except Exception as e:
            self.logger.warning(f"Context cleanup error: {e}")
        
        try:
            if self.engine:
                del self.engine
                self.engine = None
        except Exception as e:
            self.logger.warning(f"Engine cleanup error: {e}")
        
        self.logger.info("Cleanup tamamlandı")

    def __del__(self):
        """Destructor"""
        if not self._cleaned_up:
            self.cleanup()