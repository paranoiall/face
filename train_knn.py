import cv2
import json
import os
import numpy as np
from keras.models import load_model
from sklearn.model_selection import cross_val_score, ShuffleSplit, KFold
from sklearn.neighbors import KNeighborsClassifier
import joblib
# import matplotlib.pyplot as plt
from getdata import base_path

image_path = base_path


def load_data():
    images = []
    labels = []
    class_num = 0
    info = {}
    for person in os.listdir(image_path):
        path = image_path + person
        if os.path.isdir(path) is True:
            for item in os.listdir(path):
                if item.endswith('.jpg'):
                    image = cv2.imread(path + '/' + item)
                    if not image is None:
                        images.append(image)
                        labels.append(class_num)
            info[class_num] = person
            class_num += 1

    images = np.array(images, dtype='float')
    labels = np.array(labels)
    file = open(image_path + '/info.txt', 'w')
    file.write(json.dumps(info))
    file.close()
    return images, labels


def img_to_encoding(images, model):
    # image的格式是opencv读入后的格式BGR
    images = images[..., ::-1]
    # np.around是四舍五入，其中decimals是保留的小数位数,这里进行了归一化
    images = np.around(images / 255.0, decimals=12)
    if images.shape[0] > 1:
        # predict是对多个batch进行预测，这里的128是尝试后得出的内存能承受的最大值
        embedding = model.predict(images, batch_size=128)
    else:
        # predict_on_batch是对单个batch进行预测
        embedding = model.predict_on_batch(images)
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding


class Dataset:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    # 加载数据集
    def load(self):
        images, labels = load_data()
        facenet = load_model('./model/facenet.model.h5')
        # 生成128维特征向量
        X_embedding = img_to_encoding(images, facenet)
        # 输出训练集、验证集和测试集的数量
        print('X_train shape', X_embedding.shape)
        print('y_train shape', labels.shape)
        print(X_embedding.shape[0], 'train samples')
        self.X_train = X_embedding
        self.y_train = labels


class Knn_Model:
    def __init__(self):
        self.model = None

    def cross_val_and_build_model(self, dataset):
        k_range = range(1, 31)
        k_scores = []
        print("k vs accuracy:")
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            cv = ShuffleSplit(random_state=0)
            score = cross_val_score(knn, dataset.X_train, dataset.y_train, cv=cv, scoring='accuracy').mean()
            k_scores.append(score)
            # print(k, ":", score)
        # 可视化结果
        # plt.plot(k_range, k_scores)
        # plt.xlabel('Value of K for KNN')
        # plt.ylabel('Cross-Validated Accuracy')
        # plt.show()
        n_neighbors_max = np.argmax(k_scores) + 1
        print("The best k is: ", n_neighbors_max)
        print("The accuracy is: ", k_scores[n_neighbors_max - 1], "When n_neighbor is: ", n_neighbors_max)

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors_max)

    def train(self, dataset):
        self.model.fit(dataset.X_train, dataset.y_train)

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)


if __name__ == "__main__":
    dataset = Dataset()
    dataset.load()
    model = Knn_Model()
    model.cross_val_and_build_model(dataset)
    model.train(dataset)
    model.save_model('./model/face_knn.model')
