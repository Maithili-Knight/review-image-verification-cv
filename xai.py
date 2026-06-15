import numpy as np
import cv2
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image

class GradCAM:
    def __init__(self, model):
        self.model = model
        
        # In this specific architecture, the base EfficientNet is a layer within the parent model
        self.base_model = None
        for layer in self.model.layers:
            if 'efficientnet' in layer.name.lower():
                self.base_model = layer
                break
                
        if self.base_model is None:
            raise ValueError("Could not find base EfficientNet model for Grad-CAM")
            
        # We need a new model that outputs both the predictions AND the activations of the last conv layer
        # Since the base model is nested, we construct a gradient model
        
        # We need to trace from the parent model's input, through the base model, to the output
        self.grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.base_model.output, self.model.output]
        )

    def generate_heatmap(self, img_array):
        """
        Generates a Grad-CAM heatmap for the input image array.
        img_array should be preprocessed and expanded (1, 224, 224, 3)
        """
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(img_array)
            # Since this is binary classification, we just want the gradient for the class prediction
            class_channel = preds[:, 0]
            
        # Compute gradient
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Pool the gradients over spatial dimensions (ignoring batch dim)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map by the "feature importance"
        last_conv_layer_output = last_conv_layer_output[0]
        # using matrix multiplication
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def apply_heatmap(self, img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlays the given heatmap on the original image and returns a base64 string.
        """
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert to 8-bit RGB
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose
        superimposed = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
        
        # Convert to Base64
        pil_img = Image.fromarray(superimposed)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        encoded_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        
        return "data:image/jpeg;base64," + encoded_string
