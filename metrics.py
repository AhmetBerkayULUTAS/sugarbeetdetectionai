import time

class Metrics:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.image_acquisition_times = []
        self.inference_times = []

    def add_acquisition_time(self, t):
        self.image_acquisition_times.append(t)
        if len(self.image_acquisition_times) > 100:
            self.image_acquisition_times.pop(0)

    def add_inference_time(self, t):
        self.inference_times.append(t)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)

    def compute(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        avg_img_acq = sum(self.image_acquisition_times)/len(self.image_acquisition_times) if self.image_acquisition_times else 0
        avg_inf = sum(self.inference_times)/len(self.inference_times) if self.inference_times else 0
        latency = avg_img_acq + avg_inf

        return {
            "fps": fps,
            "img_acq": avg_img_acq,
            "inf": avg_inf,
            "latency": latency
        }
