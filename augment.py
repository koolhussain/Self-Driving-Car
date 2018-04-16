import os
import cv2
import matplotlib.image as mpimg
import numpy as np

def load_image(image_dir, filename):
	return mpimg.imread(os.path.join(image_dir,filename.strip()))

def choose_image(data_dir, center, left, right, steering_angle):


def augment(data_dir, center, left, right, steering_angle):
	#Choose one Image from Center, Left, Right Images
	image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
	#Vertically Flip the Image
	image, steering_angle = random_flip(image, steering_angle)
	#Shift Image Vertically and Horizontally
	image, steering_angle = random_shift(image, steering_angle, 100, 10)
	#Produce Random Shadow
	image = random_shadow(image)
	#Produce Random Brightness 
	image = random_brightness(image)
	return image, steering_angle

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
	#Creating Empty 4-D Numpy Array to store the images & Empty 1-D Numpy Array to store the steering angle
	images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
	steers = np.empty(batch_size)
	#Since No of images can be Variable so while True is used to make batches until all index are exhausted
	while True:
		i = 0 
		#Retrieving Index Values from the Images paths file
		for index in np.random.permutation(image.paths.shape[0]):
			center, left, right = image_paths[index]
			steering_angle = image_paths[index]
			if is_training and np.random.rand() < 0.6 :
				image, steering_angle = augment(data_dir, center, left, right, steering_angle)
			else:
				image = load_image(data_dir, center)
			images[i] = preprocess(image)
			steers[i] = steering_angle
			i = i + 1
			if i == batch_size:
				break
		yield images, steers


