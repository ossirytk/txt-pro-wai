from PIL import Image

#read the image
im = Image.open("src_images/english_hd_02.jpg")

#rotate image
angle = 90
out = im.rotate(angle,expand=True)
out.save("src_images/rotated_english_hd_02.jpg")
