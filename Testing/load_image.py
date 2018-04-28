import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename="image.jpg"
def load_image(filename):
	return mpimg.imread(filename.strip())

image = load_image(filename)
print(image)

plt.imshow(load_image(filename))
plt.show()