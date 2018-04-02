import numpy as np
import cv2


mat = np.float32([[11,12],[21,22]])
print(mat)
trans_m = np.float32([[1, 0, 1], [0, 1, 1]])
print(trans_m)
height, width = 2,2
print(height)
print(width)
image = cv2.warpAffine(mat, trans_m, (width, height))

print(image)