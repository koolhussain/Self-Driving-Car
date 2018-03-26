import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


image = mpimg.imread("image.jpg")

trans_x = 100*(np.random.rand() - 0.5)
print(trans_x)
trans_y = 10*(np.random.rand() - 0.5)
print(trans_y)
trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
print(trans_m)
height, width = image.shape[:2]
print(height)
print(width)
image = cv2.warpAffine(image, trans_m, (width, height))

plt.imshow(image)
plt.show()