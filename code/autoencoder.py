from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.datasets import mnist
import keras.initializers as init
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

input_img = Input(shape=(784,))

def autoencoder(input_img):
       
    # build the encoder 
    encoded = Dense(256, activation='relu', kernel_initializer = init.RandomNormal(mean=0.0, stddev=0.05, seed=None),bias_initializer=init.RandomNormal(mean=0.0, stddev=0.05, seed=None))(input_img)
    encoded = Dense(128, activation='relu', kernel_initializer = init.RandomNormal(mean=0.0, stddev=0.05, seed=None),bias_initializer=init.RandomNormal(mean=0.0, stddev=0.05, seed=None))(encoded)

    # build the decoder 
    decoded = Dense(128, activation='relu', kernel_initializer = init.RandomNormal(mean=0.0, stddev=0.05, seed=None),bias_initializer=init.RandomNormal(mean=0.0, stddev=0.05, seed=None))(encoded)
    decoded = Dense(256, activation='relu', kernel_initializer = init.RandomNormal(mean=0.0, stddev=0.05, seed=None),bias_initializer=init.RandomNormal(mean=0.0, stddev=0.05, seed=None))(decoded)
    decoded = Dense(784, activation='sigmoid', kernel_initializer = init.RandomNormal(mean=0.0, stddev=0.05, seed=None),bias_initializer=init.RandomNormal(mean=0.0, stddev=0.05, seed=None))(decoded)
    return decoded

# build the model   autoencoder 
autoencoder = Model(input_img, autoencoder(input_img))

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

epochs = 20

autoencoder_train = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# plot losses

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# predict some images
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n +1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()