import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthEstimator:
    """
    A class to perform monocular depth estimation using a pre-trained model.
    
    Attributes:
        processor (AutoImageProcessor): Pretrained image processor for model input preparation.
        model (AutoModelForDepthEstimation): Pretrained depth estimation model.
    """
    def __init__(self, model_name, device):
        """
        Initialize the DepthEstimator with the given model and device.
        
        Args:
            model_name (str): The name of the pre-trained model from the Hugging Face hub.
            device (str): The device to load the model on ('cpu' or 'cuda').
        """
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    
    def estimate_depth(self, frame_rgb):
        """
        Perform depth estimation on an input RGB image.

        Args:
            frame_input (numpy.ndarray or PIL.Image): The input image in RGB format.
        
        Returns:
            numpy.ndarray: A 2D array representing the estimated depth of each pixel in the input image.
        """
        frame_input = self.processor(images=frame_rgb, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            frame_output = self.model(**frame_input)
        return frame_output.predicted_depth.squeeze().cpu().numpy()
