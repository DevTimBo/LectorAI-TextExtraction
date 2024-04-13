import easyocr
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext('0_5.jpg')
print(result)