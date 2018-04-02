import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread("image.jpg")

HEIGHT, WIDTH = image.shape[:2]

x1, y1 = WIDTH*np.random.rand(), 0
print(x1, y1)
x2, y2 = WIDTH*np.random.rand(), HEIGHT
print(x2, y2)
xm, ym = np.mgrid[0:HEIGHT, 0:WIDTH]

mask = np.zeros_like(image[:,:,1])
mask[(ym - y1) * (x2 - x1) - (y2 - y1)*(xm - x1) > 0] = 1


#cond = mask == np.random.randint(2) 
cond = mask == np.random.randint(2)
s_ratio = np.random.uniform(low=0.2, high=0.5)

hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
hls[:,:,1][cond] = hls[:,:,1][cond]*s_ratio
shadow = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

cond = mask == np.random.randint(2)
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
hls[:,:,1][cond] = hls[:,:,1][cond]*s_ratio
shadow_mask = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


# plt.imshow(image)
# plt.imshow(hls)
# plt.imshow(shadow)
# plt.show()

cv2.imshow("HLS", hls)
cv2.imshow("Shadow Image", cv2.cvtColor(shadow, cv2.COLOR_BGR2RGB))
cv2.imshow("Shadow_mask Image", cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2RGB))
cv2.imshow("Original_Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()