from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import svm
from xgboost import XGBClassifier
import numpy as np
import joblib
import time
import cv2
import os



class EmotionDetection:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.model = None
        
        self.X_train = np.empty((0,2304))
        self.y_train = np.array([])
        self.X_test = np.empty((0,2304))
        self.y_test = np.array([])

        self.X_pca = None

    def load_cnn_model(self, path='model.json'):
        jfile = open(path, 'r')
        jmodel = jfile.read()
        jfile.close()
        return model_from_json(jmodel)

    def visual_test_model(self, video_path, webcam=False, cnn=True):
        emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        if cnn:
            self.model = self.load_cnn_model()
            self.model.load_weights("model.h5")

        if webcam:
            v = cv2.VideoCapture(0)
        else:
            v = cv2.VideoCapture(video_path)

        while True:
            ret, frame = v.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(grayscale_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 0, 255), 2)
                gray_face = grayscale_frame[y:y + h, x:x + w]
                cropped = np.expand_dims(np.expand_dims(cv2.resize(gray_face, (48, 48)), -1), 0)
                
                if cnn:
                    idx = int(np.argmax(self.model.predict(cropped)))
                else:
                    idx = int(self.model.predict(cropped.flatten().reshape(1, -1)))
                
                cv2.putText(frame, emotions[idx], (x+5, y+270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        v.release()
        cv2.destroyAllWindows()

    def get_model_accuracy(self, pca=False, n_components=3):
        if self.model != None:
            if pca:
                pca = PCA(n_components=n_components)
                self.X_pca = pca.fit_transform(self.X_test)

            print("start: testing...")
            predictions = self.model.predict(self.X_pca if pca else self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            print("finished: testing")

            print("ACCURACY: ", accuracy)
    
    def load_model(self, path='svm_model.pkl'):
        self.model = joblib.load(path)

    def load_data(self):
        print("start: loading training data...")
        for idx, i in enumerate(self.emotions):
            for j in os.listdir('./data/train/{0}'.format(i)):
                self.X_train = np.vstack([self.X_train, cv2.cvtColor(cv2.imread('./data/train/{0}/{1}'.format(i, j), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).flatten()])
                self.y_train = np.append(self.y_train, idx)
        print("finished: loading training data")

        print("start: loading testing data...")
        for idx, i in enumerate(self.emotions):
            for j in os.listdir('./data/test/{0}'.format(i)):
                self.X_test = np.vstack([self.X_test, cv2.cvtColor(cv2.imread('./data/test/{0}/{1}'.format(i, j), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).flatten()])
                self.y_test = np.append(self.y_test, idx)
        print("finished: loading testing data")

    def cnn_model_training(self):
        train_data_gen = ImageDataGenerator(rescale=1./255)
        validation_data_gen = ImageDataGenerator(rescale=1./255)

        train_generator = train_data_gen.flow_from_directory(
            self.train_path,
            target_size=(48, 48),
            batch_size=64,
            color_mode='grayscale',
            class_mode='categorical'
        )

        validation_generator = validation_data_gen.flow_from_directory(
            self.test_path,
            target_size=(48, 48),
            batch_size=64,
            color_mode='grayscale',
            class_mode='categorical'
        )

        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=0.0001, decay=0.000001), metrics=['accuracy'])

        model.fit(
            train_generator,
            steps_per_epoch=2800 // 64,
            epochs=100,
            validation_data=validation_generator,
            validation_steps=700 // 64)
        
        self.model = model

        jmodel = model.to_json()
        with open("model.json", "w") as jfile:
            jfile.write(jmodel)

        model.save_weights('model.h5')

    def svm_model_training(self, pca=False, C=1, kernel='rbf', degree=3, gamma='scale', n_components=3):
        if pca:
            pca = PCA(n_components=n_components)
            self.X_pca = pca.fit_transform(self.X_train)
        
        print("start: fitting...")
        start = time.time()
        self.model = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma).fit(self.X_pca if pca else self.X_train, self.y_train)
        print(time.time() - start)
        print("finished: fitting")

        joblib.dump(self.model, 'svm_model.pkl')  

    def xgb_model_training(self, pca=False, gamma=0, eta=0.3, max_depth=6, n_components=3):
        if pca:
            pca = PCA(n_components=n_components)
            self.X_pca = pca.fit_transform(self.X_train)

        print("start: fitting...")
        start = time.time()
        self.model = XGBClassifier(gamma=gamma, eta=eta, max_depth=max_depth).fit(self.X_pca if pca else self.X_train, self.y_train)
        print(time.time() - start)
        print("finished: fitting")

        joblib.dump(self.model, 'xgb_model.pkl')  

    def rfc_model_training(self, pca=False, n_estimators=100, max_depth=None, n_components=3):
        if pca:
            pca = PCA(n_components=n_components)
            self.X_pca = pca.fit_transform(self.X_train)

        print("start: fitting...")
        start = time.time()
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth).fit(self.X_pca if pca else self.X_train, self.y_train)
        print(time.time() - start)
        print("finished: fitting")

        joblib.dump(self.model, 'rfc_model.pkl') 

ed = EmotionDetection('data/train', 'data/test')
ed.load_data()

ed.svm_model_training()
ed.get_model_accuracy()

ed.xgb_model_training()
ed.get_model_accuracy()

ed.rfc_model_training()
ed.get_model_accuracy()

ed.cnn_model_training()
ed.visual_test_model("./vid2.mp4", cnn=True)