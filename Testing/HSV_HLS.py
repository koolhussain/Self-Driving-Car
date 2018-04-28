import cv2

image = cv2.imread("image.jpg")
hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# print("Original Image")
# print(image)

# print("HLS Image")
# print(hls_image)

# print("HSV Image")
# print(hsv_image)


image_R = image[:,:,0]
image_G = image[:,:,1]
image_B = image[:,:,2]

hls_image_H = hls_image[:,:,0]
hls_image_L = hls_image[:,:,1]
hls_image_S = hls_image[:,:,2]


# print(image[:,:,1].shape)
print(hls_image.shape)
print(hsv_image.shape)

cv2.imshow("image_R", image_R)
cv2.imshow("image_G", image_G)
cv2.imshow("image_B", image_B)

cv2.imshow("hls_image_H", hls_image_H)
cv2.imshow("hls_image_L", hls_image_L)
cv2.imshow("hls_image_S", hls_image_S)

# cv2.imshow("hls_image", hls_image)
# cv2.imshow("hsv_image", hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()