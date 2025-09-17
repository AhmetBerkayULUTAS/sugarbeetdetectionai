import cv2

class Visualizer:
    def __init__(self, class_names):
        self.class_names = class_names

    def draw(self, frame, results, metrics):
        annotated = frame.copy() 
        for r in results:
            annotated = r.plot(img=annotated)

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
