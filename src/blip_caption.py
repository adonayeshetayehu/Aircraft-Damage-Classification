from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tensorflow as tf

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, processor, model, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        return tf.py_function(self.process_image, [image_path, task], tf.string)

    def process_image(self, image_path, task):
        # Convert bytes to string
        image_path_str = image_path.numpy().decode("utf-8")
        task_str = task.numpy().decode("utf-8")

        # Open image
        image = Image.open(image_path_str).convert("RGB")

        # Select prompt
        prompt = "This is a picture of" if task_str == "caption" else "This is a detailed photo showing"

        # Add padding and truncation to avoid tensor shape issues
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Generate caption
        output = self.model.generate(**inputs)

        # Decode to string
        caption_text = self.processor.decode(output[0], skip_special_tokens=True)

        # Return as tf.Tensor string
        return tf.constant(caption_text, dtype=tf.string)

def generate_text(image_path, task):
    blip_layer = BlipCaptionSummaryLayer(processor, blip_model)
    return blip_layer(tf.constant(image_path), tf.constant(task))