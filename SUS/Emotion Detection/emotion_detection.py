import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import cv2



class EmotionDetection:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_model(self, path='model.json'):
        jfile = open(path, 'r')
        jmodel = jfile.read()
        jfile.close()
        return model_from_json(jmodel)

    def train_model(self):
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

        jmodel = model.fit_generator(
            train_generator,
            steps_per_epoch=28709 // 64,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=7178 // 64)

        jmodel = jmodel.to_json()
        with open("model.json", "w") as jfile:
            jfile.write(jmodel)

        model.save_weights('model.h5')

    def test_model(self, video_path, webcam=False):
        emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        model = self.load_model()
        model.load_weights("model.h5")

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

                cv2.putText(frame, emotions[int(np.argmax(model.predict(cropped)))], (x+5, y+270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        v.release()
        cv2.destroyAllWindows()


ed = EmotionDetection('data/train', 'data/test')
ed.test_model("C:\\Users\\Maksim\\Desktop\\repos\\PUT-WI-SI\\SUS\\Emotion Detection\\vid2.mp4")