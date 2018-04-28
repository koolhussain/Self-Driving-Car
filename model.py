import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from augment import INPUT_SHAPE, batch_generator

data_dir = os.path.join(os.getcwd(), "IMG")
TEST_SIZE = 0.2
KEEP_PROB = 0.5
LR = 0.0001
BATCH_SIZE = 32
SAMPLE_PER_EPOCH = 20000
N_EPOCHS = 10

def load_data(data_dir, test_size):
	data_df = pd.read_csv("driving_log.csv", names=["center","left","right","steering_angle","throttle","reverse","speed"])

	# print(data_df.head())
	# print(data_df.tail())
	# print(data_df.iloc[0])
	# print(data_df.iloc[1][1])

	# print(data_df.iloc[0]["center"])
	# print(data_df.iloc[1][1])

	X = data_df[["center", "left", "right"]].values
	# print(X)
	y = data_df["steering_angle"].values
	# print(y)

	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=TEST_SIZE)
	# print(X_train[0])
	# print(X_valid[0])
	# print(y_train[0])
	# print(y_valid[0])
	return X_train, X_valid, y_train, y_valid

def build_model():
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=INPUT_SHAPE))
	model.add(Conv2D(24, (5,5), activation='elu', strides=(2,2)))
	model.add(Conv2D(36, (5,5), activation='elu', strides=(2,2)))
	model.add(Conv2D(48, (5,5), activation='elu', strides=(2,2)))
	model.add(Conv2D(48, (3,3), activation='elu'))
	model.add(Conv2D(48, (3,3), activation='elu'))
	model.add(Flatten())
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))
	model.summary()
	return model

def train_model(model, LR, BATCH_SIZE, SAMPLE_PER_EPOCH, N_EPOCHS, data_dir, X_train, X_valid, y_train, y_valid):
	checkpoint = ModelCheckpoint('model-{epoch:03d}-{val_loss:.02f}.h5',
		monitor='val_loss',
		verbose=0,
		save_best_only=True,
		mode='auto')
	tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
	model.compile(loss='mean_squared_error', optimizer=Adam(lr=LR))
	model.fit_generator(batch_generator(data_dir, X_train, y_train, BATCH_SIZE, True),
		SAMPLE_PER_EPOCH, N_EPOCHS,
		validation_data=batch_generator(data_dir, X_valid, y_valid, BATCH_SIZE, False),
		callbacks=[checkpoint, tensorboard], verbose=1, max_queue_size=1, validation_steps=len(X_valid))
	score = model.evaluate(X_valid, y_valid)
	print(score)

if __name__ == '__main__':
	data = load_data(data_dir, TEST_SIZE)
	model = build_model()
	train_model(model, LR, BATCH_SIZE, SAMPLE_PER_EPOCH, N_EPOCHS, data_dir, *data)





	


