import easyocr
reader = easyocr.Reader(['de'], gpu=False)
result = reader.readtext('0_5.jpg')
print(result)