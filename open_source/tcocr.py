from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

image = Image.open("0_5.jpg").convert("RGB")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, max_new_tokens=93)[0]
print(generated_text)