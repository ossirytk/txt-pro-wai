import pytesseract
text = pytesseract.image_to_string(Image.open("processed_image.png"))
print(text)
