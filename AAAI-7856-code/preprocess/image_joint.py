from PIL import Image

# Open the three images
image1 = Image.open("D:\\data\\1.png")
image2 = Image.open("D:\\data\\2.png")
image3 = Image.open("D:\\data\\3.png")

# Resize the images to 224x224
image1 = image1.resize((224, 224))
image2 = image2.resize((224, 224))
image3 = image3.resize((224, 224))
# Create a new image with a size of 672x224
new_image = Image.new('RGB', (672, 224))

# Paste the three images into the new image
new_image.paste(image1, (0, 0))
new_image.paste(image2, (224, 0))
new_image.paste(image3, (448, 0))

new_image_r = new_image.resize((224, 224))

# Save the new image
new_image_r.save("D:\\data-joint\\1.png")