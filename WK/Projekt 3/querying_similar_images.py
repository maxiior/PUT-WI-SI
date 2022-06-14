from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Dense, Reshape, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
import os
from keras.applications import VGG16
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.decomposition import PCA
import random
from scipy.spatial import distance


class QueryingSimilarImages:
    # W konstruktorze zmienne height i width mówią do jakiego rozmiaru będziemy reskalować obrazy wchodzące do sieci cnn, a batch to rozmiar kubełka w sieci cnn,
    # którego wyniki będą uśredniane
    def __init__(self, data_path, height=48, width=48, batch=16):
        self.model = None
        self.height = height
        self.width = width
        self.batch = batch
        self.features = None
        self.images = None

        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        # Wczytywanie danych traningowych podzielonych na kategorie względem folderów
        self.training_set = datagen.flow_from_directory(
            data_path,
            target_size=(height, width),
            batch_size=batch,
            class_mode='input',
            subset='training',
            shuffle=True)

        # Wczytywanie danych walidacyjnych podzielonych na kategorie względem folderów
        self.validation_set = datagen.flow_from_directory(
            data_path,
            target_size=(height, width),
            batch_size=batch,
            class_mode='input',
            subset='validation',
            shuffle=False)

    # Funkcja służąca do trenowania modelu na zbiorze danych podanym w konstruktorze
    def train_model(self, epochs=100):
        # Architektura sieci została zainspirowana jednym z artykułów znajdujących się w Bibliografii
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

    # Wczytywanie naszego własnego wytrenowanego modelu
    def load_own_model(self, file='model.h5'):
        self.model = load_model(file)

    # Funkcja przeznaczona do wczytywania pojedynczego zdjęcia
    def _load_image(self, path):
        img = image.load_img(path, target_size=self.model.input_shape[1:3])
        return img, preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

    # Funkcja odpowiedzialna za znajdowanie najbardziej podobnych zdjęć, korzystając z naszego własnego wytrenowanego modelu
    def find_similar_photos_by_own_trained_model(self, path=None, n_components=100):
        # Jeżeli nie podaliśmy ścieżki to jest ona losowana spośród dostępnych w self.images
        if path == None:
            idx = int(len(self.images) * random.random())
            path = self.images[idx]
        else:
            idx = len(self.images)
            self.images.append(path)

        # Wykonujemy ekstrakcję cech na zdjęciach, używamy procedury PCA w celu ograniczenia zbędnych informacji i porównujemy odległość cosinusową pomiędzy wektorami
        # cech zdjęć
        features = []
        for i, path in enumerate(self.images):
            if i % int(len(self.images)/10) == 0:
                print("{0}/{1}".format(i, len(self.images)))
            _, x = self._load_image(path)
            features.append(np.average(np.average(
                self.model.predict(x)[0], axis=2), axis=0))
        features = np.array(features)
        pca = PCA(n_components=n_components)
        pca.fit(features)
        self.features = pca.transform(features)

        similar = [distance.cosine(self.features[idx], i)
                   for i in self.features]
        closest = sorted(range(len(similar)),
                         key=lambda k: similar[k])[1:7]

        # Wyznaczamy zdjęcia najbliższe temu zapisanego w path i wyświetlamy 5 najlepszych wyników
        thumbs = []
        for i in closest:
            img = image.load_img(self.images[i])
            img = img.resize((int(img.width * 100 / img.height), 100))
            thumbs.append(img)

        images = np.concatenate([np.asarray(t) for t in thumbs], axis=1)

        plt.figure(figsize=(16, 12))
        plt.title('Względem pierwszego zdjęcia zostało znalezione pięć kolejnych')
        plt.imshow(images)
        plt.show()

    # Funkcja wczytująca pretrenowany model imagenet
    def load_pretrained_model(self):
        self.model = VGG16(weights='imagenet', include_top=True)

    # Funkcja wczytująca ścieżki do poszczególnych zdjęć
    def load_images_paths(self, path="./ALL/"):
        self.images = [os.path.join(dp, f) for dp, _, filenames in os.walk(
            path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]

    # Funkcja przeznaczona do ekstrakcji cech ze wczytanych zdjęć
    def _extracting_features(self):
        extractor = Model(inputs=self.model.input,
                          outputs=self.model.get_layer("fc2").output)

        print("keeping {0} images to analyze".format(len(self.images)))

        # Wykonujemy ekstrakcję cech na zdjęciach, używamy procedury PCA w celu ograniczenia zbędnych informacji i porównujemy odległość cosinusową pomiędzy wektorami
        # cech zdjęć
        features = []
        for i, path in enumerate(self.images):
            if i % int(len(self.images)/10) == 0:
                print("{0}/{1}".format(i, len(self.images)))
            _, x = self._load_image(path)
            features.append(extractor.predict(x)[0])

        print('finished extracting features for {0} images'.format(
            len(self.images)))

        features = np.array(features)
        pca = PCA(n_components=100)
        pca.fit(features)

        self.features = pca.transform(features)

    # Funkcja odpowiedzialna za znajdowanie najbardziej podobnych zdjęć, korzystając z pretrenowanego modelu
    def find_similar_photos_by_pretrained_model(self, path=None):
        # Jeżeli nie podaliśmy ścieżki to jest ona losowana spośród dostępnych w self.images
        if path == None:
            idx = int(len(self.images) * random.random())
            path = self.images[idx]
        else:
            idx = len(self.images)
            self.images.append(path)

        self._extracting_features()

        similar = [distance.cosine(self.features[idx], i)
                   for i in self.features]
        closest = sorted(range(len(similar)),
                         key=lambda k: similar[k])[1:7]

        # Wyznaczamy zdjęcia najbliższe temu zapisanego w path i wyświetlamy 5 najlepszych wyników
        thumbs = []
        for i in closest:
            img = image.load_img(self.images[i])
            img = img.resize((int(img.width * 100 / img.height), 100))
            thumbs.append(img)

        images = np.concatenate([np.asarray(t) for t in thumbs], axis=1)

        self.images = self.images[:-1]

        plt.figure(figsize=(16, 12))
        plt.title('Względem pierwszego zdjęcia zostało znalezione pięć kolejnych')
        plt.imshow(images)
        plt.show()


path = "./images/mikhail-vasilyev-130018-unsplash.jpg"

# W folderze data mieliśmy 10 folderów przechowujących 10 kategorii zdjęć
qsi = QueryingSimilarImages("./data", height=256, width=256)
# W folderze images mieliśmy wszystkie zdjęcia ze wszystkich kategorii połączone ze sobą w jeden zbiór
qsi.load_images_paths(path="./images/")

# Wyznaczanie najbardziej podobnych zdjęć za pomocą pretrenowanego modeli imagenet
qsi.load_pretrained_model()
qsi.find_similar_photos_by_pretrained_model(path=path)

# Wyznaczanie najbardziej podobnych zdjęć za pomocą naszego wyszkolonego modelu
qsi.load_own_model()
qsi.find_similar_photos_by_own_trained_model(path=path)


# Korzystaliśmy ze zbioru zdjęć animals10, który zawiera około 30.000 zdjęć zwierząt w 10 kategoriach. Okroiliśmy ten zbiór do dwóch podzbiorów składających się z 2.000 zdjęć i 200 zdjęć,
# aby ograniczyć czas uczenia modeli na tych zbiorach. W przypadku naszych podzbiorów, poszczególne kategorie składały się z równej liczby zdjęć.

# Wnioski
# Pretrenowany model imagenet sprawdza się świetnie w poszukiwaniach najbliższych zdjęć i niemal w każdym przypadku podaje 5/5 zdjęć należących do tej samej kategorii co zdjęcie
# dane na wejściu. W przypadku naszych wytrenowanych modeli na zdjęciach przeskalowanych do rozmiarów 24x24, 100x100 i 256x256, w każdym przypadku wyniki nie są najlepsze i często
# model podaje tylko 2/5 lub 3/5 zdjęć z tej samej kategorii. Prawdopodobnie wynika to z braku odpowiednich zasobów, aby wystarczająco skutecznie wyszkolić daną sieć.


# Bibliografia
# https://www.kaggle.com/datasets/alessiocorrado99/animals10
# https://www.analyticsvidhya.com/blog/2021/01/querying-similar-images-with-tensorflow/
# https://www.youtube.com/watch?v=aACebENHlrs
