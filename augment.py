import os
import cv2
import matplotlib.image as mpimg
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def preprocess(image):
	image = crop(image)
	image = resize(image)
	image = rgb2yuv(image)
	return image
	
def crop(image):
	return image[60: -25, :, :]

def resize(image):
	return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def load_image(image_dir, filename):
	return mpimg.imread(os.path.join(image_dir,filename.strip()))

def random_flip(image, steering_angle):
	if np.random.rand() < 0.5:
		image = cv2.flip(image, 1)
		steering_angle = -steering_angle
	return image, steering_angle

def random_shift(image, steering_angle, x_range, y_range):
	trans_x = x_range*(np.random.rand() - 0.5)
	trans_y = y_range*(np.random.rand() - 0.5)
	steering_angle = steering_angle + trans_x*0.02
	trans_m = np.float32([[1,0,trans_x],[0,1,trans_y]])
	HEIGHT, WIDTH = image.shape[:2]
	image = cv2.warpAffine(image, trans_m, (WIDTH, HEIGHT))
	return image, steering_angle

def random_shadow(image):
	x1, y1 = IMAGE_WIDTH*np.random.rand(), 0
	x2, y2 = IMAGE_WIDTH*np.random.rand(), IMAGE_HEIGHT
	xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
	mask = np.zeros_like(image[:,:,1])
	mask[(ym - y1)*(x2 - x1)-(y2 - y1)*(xm - x1)>0]=1
	cond=mask==np.random.randint(2)
	s_ratio = np.random.uniform(low=0.2, high=0.5)
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	hls[:,:,1][cond] = hls[:,:,1][cond]*s_ratio
	return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	ratio = 1.0 + 0.4*(np.random.rand() - 0.5)
	hsv[:,:,2] = hsv[:,:,2]*ratio
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def choose_image(data_dir, center, left, right, steering_angle):
	choice = np.random.choice(3)
	if choice == 0:
		return load_image(data_dir, left), steering_angle + 0.2
	elif choice == 1:
		return load_image(data_dir, right), steering_angle - 0.2
	return load_image(data_dir, center), steering_angle

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
		for index in np.random.permutation(image_paths.shape[0]):
			center, left, right = image_paths[index]
			steering_angle = steering_angles[index]
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


