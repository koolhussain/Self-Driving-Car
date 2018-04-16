import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename="image.jpg"
def load_image(filename):
	return mpimg.imread(filename.strip())

print(load_image(filename))

plt.imshow(load_image(filename))
plt.show()