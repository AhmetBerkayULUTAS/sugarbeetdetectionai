import cv2
import numpy as np # Manuel çizim için gerekli

class Visualizer:
    def __init__(self, class_names):
        self.class_names = class_names
        # çoklu sınıf tespitleri için rastgele bir renk oluşturma
        #np.random.seed(42) 
        #self.colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")
        
        # Tek sınıf için sabit bir renk
        self.colors = np.array([[0, 255, 0]], dtype="uint8")

    def draw(self, frame, results, metrics):
        annotated = frame.copy() 
        
        
        for detection in results:
            box = detection['box']
            score = detection['score']
            class_id = detection['class_id']
            
            # Kutu Koordinatları (x1, y1, x2, y2)
            x1, y1, x2, y2 = box
            
            # Renk ve Sınıf Adı
            color = [int(c) for c in self.colors[class_id]]
            class_name = self.class_names[class_id]
            label = f"{class_name}: {score:.2f}"

            # Kutuyu Çizme
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Etiket için arka planı çizme
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            
            # Etiketi kutunun üzerine yazma
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Metrikleri Çizme 
        if annotated is not None:
            cv2.putText(annotated, f"FPS {metrics['fps']:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"Image Acquisition {metrics['img_acq']:.2f} ms", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"Inference {metrics['inf']:.2f} ms", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"Latency {metrics['latency']:.2f} ms", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return annotated