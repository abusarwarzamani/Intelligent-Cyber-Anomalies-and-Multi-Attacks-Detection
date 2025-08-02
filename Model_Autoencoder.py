import keras
from keras import layers
import cv2 as cv
import numpy as np

from Evaluation import evaluation


# from Evaluate_Error import evaluation


def Model_AutoEncoder(Train_Data, Train_Target, Test_Data, Test_Target, sol=2):
    # This is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(Train_Data.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(Train_Target.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    Train_Data = Train_Data.astype('float32') / 255.
    Test_Data = Test_Data.astype('float32') / 255.
    Train_Data = Train_Data.reshape((len(Train_Data), np.prod(Train_Data.shape[1:])))
    Test_Data = Test_Data.reshape((len(Test_Data), np.prod(Test_Data.shape[1:])))
    autoencoder.fit(Train_Data, Train_Target,
                    epochs=int(sol),
                    batch_size=256,
                    shuffle=True,
                    validation_data=(Test_Data, Test_Target))
    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs = encoder.predict(Test_Data)
    pred = decoder.predict(encoded_imgs)

    Eval = evaluation(pred, Test_Target)

    return Eval, pred


def Model_AutoEn(Data, Target, sol=None):
    if sol is None:
        sol = [128,1,0.01]
    # This is the size of our encoded representations
    encoding_dim = Data.shape[0]  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(Data.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(Target.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    Train_Data = Data.astype('float32') / 255.
    Test_Data = Data.astype('float32') / 255.
    Train_Data = Train_Data.reshape((len(Train_Data), np.prod(Train_Data.shape[1:])))
    Test_Data = Test_Data.reshape((len(Test_Data), np.prod(Test_Data.shape[1:])))
    autoencoder.fit(Train_Data, Target,
                    epochs=int(sol[1]),
                    batch_size=256,
                    shuffle=True,
                    validation_data=(Test_Data, Target))
    Weights = autoencoder.layers[-1].get_weights()[0]
    Feature = np.resize(Weights, (Data.shape[0], 100))
    return Feature
