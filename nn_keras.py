import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import load_model
from keras import optimizers
from keras import utils

# parameters
inputs, hiddens, outputs = 784, 100, 10
learning_rate = 0.01
epochs = 50
batch_size = 20

# loading datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 784).astype('float32')/255
train_labels = utils.to_categorical(train_labels, outputs)
test_images = test_images.reshape(10000, 784).astype('float32')/255
test_labels = utils.to_categorical(test_labels, outputs)

# training with the train dataset
def train():
    model = Sequential()
    model.add(Dense(hiddens, activation='sigmoid', input_shape=(inputs,)))
    model.add(Dense(outputs, activation='sigmoid'))
    sgd = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)
    model.save('mlp_model.h5')

# predicting the test dataset
def predict():
    model = load_model("mlp_model.h5")
    error = model.evaluate(test_images, test_labels)
    print("accuracy:", 1 - error)

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='predict' )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.action == "predict":
        predict()
    if FLAGS.action == "train":
        train()
