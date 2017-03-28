from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from keras.preprocessing import image
import numpy as np

train_list = glob.glob("train/*")
cat_list = glob.glob("train/*")
dog_list = glob.glob("train/*")

def img_to_tensor(file_path):
    """
    converts an image to a tensor usable by keras. 
    
    See here: https://keras.io/applications/
    """
    img = image.load_img(file_path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

import glob
cat_list = glob.glob("data/cat/*")
dog_list = glob.glob("data/dog/*")

cat_tensor = np.vstack([img_to_tensor(x) for x in cat_list])
dog_tensor = np.vstack([img_to_tensor(x) for x in dog_list])

X_train = np.vstack((cat_tensor, dog_tensor))
y_train = np.concatenate((np.zeros(cat_tensor.shape[0]), np.ones(dog_tensor.shape[0])))

# normalising
X_train /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 2)

# Now we will build a CNN. To understand CNN consider: http://cs231n.github.io/convolutional-networks/
# 
# We  need three components:
# 
# 1.  Convolution Layer - this performs the convolution filters
# 2.  Pooling - this performs the downsampling operation
# 3.  Fully connected layer - this is the normal dense neural network layer

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
input_shape = X_train.shape[1:]
# input_shape = (28, 28, 1)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], 
                       border_mode='valid', 
                       input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten()) # this is so that the last layer will be able to perform softmax
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=5, nb_epoch=10,
          verbose=1)
score = model.evaluate(X_train, Y_train, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])



