import cv2

image = cv2.imread("image.jpg")

image_flipped = cv2.flip(image, 1)

cv2.imshow("Image Flipped", image_flipped)
cv2.imshow("Original_Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
