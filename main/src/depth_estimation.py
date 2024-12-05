import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthEstimator:
    def __init__(self, model_name, device):
        print(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    
    def estimate_depth(self, frame_rgb):
        frame_input = self.processor(images=frame_rgb, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            frame_output = self.model(**frame_input)
        return frame_output.predicted_depth.squeeze().cpu().numpy()
