import cv2
import time
import json
import joblib
import os
import numpy as np
from keras.models import load_model
from getdata import image_size
from train_knn import img_to_encoding

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
classifier = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt2.xml')  # Haar分类器

fps = 5  # FPS
print('Preparing...')


class Model_cnn:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, image):
        # reshape为符合输入要求的尺寸
        image = image.reshape((1, image_size, image_size, 3))
        image = image.astype('float32')
        image /= 255
        result = self.model.predict(image)
        ID = result.argmax(axis=-1)
        accu = max(result.tolist()[0])
        return ID[0], accu


facenet = load_model('./model/facenet.model.h5')  # facenet模型（Knn）


class Model_knn:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, image):
        image_embedding = img_to_encoding(np.array([image]), facenet)
        # predict方法返回值是array of shape [n_samples]，因此下面要用label[0]从array中取得数值.
        label = self.model.predict(image_embedding)
        return label[0], 1


# model = Model_cnn('./model/face_cnn.model.h5')  # Cnn
model = Model_knn('./model/face_knn.model')  # Knn

cv2.namedWindow('Detecting your face.', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

file = open('./data/info.txt', 'r')
js = file.read()
info = json.loads(js)
file.close()

print('Start.')
time0 = time.time()
while cap.isOpened():
    if time.time() - time0 < 1 / fps:
        continue
    time0 = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(64, 64))
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            x0 = int(x + w / 8)
            x1 = int(x - w / 8) + w

            # 模型识别脸部图像
            image = cv2.resize(frame[y:y + h, x0:x1], (image_size, image_size))
            if image is None:
                break
            else:
                faceID, accu = model.predict(image)
                cv2.rectangle(frame, (x0 - 5, y - 5), (x1 + 5, y + h + 5), (255, 0, 0), 2)
                name = info[str(faceID)]
                cv2.putText(frame, name, (x0 + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detecting your face.", frame)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
