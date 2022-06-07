from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Dense, Reshape, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pickle
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd


class QueryingSimilarImages:
    def __init__(self, data_path, height=48, width=48, batch=16):
        self.model = None
        self.data_path = data_path
        self.height = height
        self.width = width
        self.batch = batch

        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        self.training_set = datagen.flow_from_directory(
            self.data_path,
            target_size=(height, width),
            batch_size=batch,
            class_mode='input',
            subset='training',
            shuffle=True)

        self.validation_set = datagen.flow_from_directory(
            self.data_path,
            target_size=(height, width),
            batch_size=batch,
            class_mode='input',
            subset='validation',
            shuffle=False)

    def eucledian_distance(self, x, y):
        eucl_dist = np.linalg.norm(x - y)
        return eucl_dist

    def load_model(self, file='model.h5'):
        self.model = load_model(file)

    def train_model(self, epochs=100):
        model = Input(shape=(self.height, self.width, 3))

        encoder = Conv2D(32, (3, 3), padding='same',
                         kernel_initializer='normal')(model)
        encoder = LeakyReLU()(encoder)
        encoder = BatchNormalization(axis=-1)(encoder)
        encoder = Conv2D(64, (3, 3), padding='same',
                         kernel_initializer='normal')(encoder)
        encoder = LeakyReLU()(encoder)
        encoder = BatchNormalization(axis=-1)(encoder)
        encoder = Conv2D(64, (3, 3), padding='same',
                         kernel_initializer='normal')(model)
        encoder = LeakyReLU()(encoder)
        encoder = BatchNormalization(axis=-1)(encoder)

        encoder_dim = K.int_shape(encoder)
        encoder = Flatten()(encoder)

        latent_space = Dense(16, name='latent_space')(encoder)

        decoder = Dense(np.prod(encoder_dim[1:]))(latent_space)
        decoder = Reshape(
            (encoder_dim[1], encoder_dim[2], encoder_dim[3]))(decoder)
        decoder = Conv2DTranspose(64, (3, 3), padding='same',
                                  kernel_initializer='normal')(decoder)
        decoder = LeakyReLU()(decoder)
        decoder = BatchNormalization(axis=-1)(decoder)
        decoder = Conv2DTranspose(64, (3, 3), padding='same',
                                  kernel_initializer='normal')(decoder)
        decoder = LeakyReLU()(decoder)
        decoder = BatchNormalization(axis=-1)(decoder)
        decoder = Conv2DTranspose(32, (3, 3), padding='same',
                                  kernel_initializer='normal')(decoder)
        decoder = LeakyReLU()(decoder)
        decoder = BatchNormalization(axis=-1)(decoder)
        decoder = Conv2DTranspose(3, (3, 3), padding="same")(decoder)

        output = Activation('sigmoid', name='decoder')(decoder)

        autoencoder = Model(model, output, name='autoencoder')
        autoencoder.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        autoencoder.fit(
            self.training_set,
            steps_per_epoch=self.training_set.n // self.batch,
            epochs=epochs,
            validation_data=self.validation_set,
            validation_steps=self.validation_set.n // self.batch,
            callbacks=[ModelCheckpoint('model.h5',
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=False)])

        self.model = Model(autoencoder.input,
                           autoencoder.get_layer('latent_space').output)

    def get_embeddings(self, file='model.h5'):
        if self.model == None:
            self.load_model(file)
        else:
            self.train_model()

        X = []
        indices = []

        for i in tqdm(range(len(os.listdir('./ALL')))):
            try:
                img_name = os.listdir('./ALL')[i]
                img = load_img('./ALL/{}'.format(img_name),
                               target_size=(self.width, self.height))
                img = img_to_array(img) / 255.0
                img = np.expand_dims(img, axis=0)
                pred = self.model.predict(img)
                pred = np.resize(pred, (16))
                X.append(pred)
                indices.append(img_name)

            except Exception as e:
                print(img_name)
                print(e)

        # Export the embeddings
        embeddings = {'indices': indices, 'features': np.array(X)}
        pickle.dump(embeddings,
                    open('./image_embeddings.pickle', 'wb'))

    def calculate_similarity(self, img_name, file='model.h5'):
        if self.model == None:
            self.load_model(file)
        else:
            self.train_model()
            self.get_embeddings()

        img = load_img('./ALL/{}'.format(img_name),
                       target_size=(self.height, self.width))
        embeddings = pickle.load(open('image_embeddings.pickle', 'rb'))

        img_similarity = []

        # Get actual image embedding
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img)
        pred = np.resize(pred, (16))

        # Calculate vectors distances
        for i in tqdm(range(len(embeddings['indices']))):
            img_name = embeddings['indices'][i]
            dist = self.eucledian_distance(pred, embeddings['features'][i])
            img_similarity.append(dist)

        imgs_result = pd.DataFrame(
            {'img': embeddings['indices'], 'euclidean_distance': img_similarity})

        imgs_result = imgs_result.query('euclidean_distance > 0').sort_values(
            by='euclidean_distance', ascending=True).reset_index(drop=True)

        for i in range(10):
            image = load_img('./ALL/{}'.format(imgs_result['img'].values[i]))
            plt.imshow(image)
            plt.show()
            print('Euclidean Distance: {}'.format(
                imgs_result['euclidean_distance'].values[i]))


qsi = QueryingSimilarImages("./images", height=256, width=256)
qsi.train_model(epochs=50)
# qsi.get_embeddings()
# qsi.calculate_similarity(
#     "OIP-Wa4llvzpHTyXgVb40_r02AHaGL.jpeg")

# https://www.kaggle.com/datasets/alessiocorrado99/animals10
# https://www.analyticsvidhya.com/blog/2021/01/querying-similar-images-with-tensorflow/
