import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image_path = "0_5.jpg"
image = Image.open(image_path)
extracted_text = pytesseract.image_to_string(image).strip()
print(extracted_text)