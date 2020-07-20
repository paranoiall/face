import cv2
import os
import random
import json
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from getdata import image_size, base_path

image_path = base_path
class_num = 0


def load_data():
    global class_num
    images = []
    labels = []
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
    return images, labels, class_num


class Dataset:
    def __init__(self):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

        # 当前库采用的维度顺序，包括rows，cols，channels，用于后续卷积神经网络模型中第一层卷积层的input_shape参数
        self.input_shape = None

    # 加载数据集
    def load(self, img_rows=image_size, img_cols=image_size, img_channels=3):
        images, labels, nb_classes = load_data()

        # 随机划分数据集
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                random_state=random.randint(0, 100))

        # tensorflow 作为后端，数据格式约定是channel_last，与这里数据本身的格式相符，如果是channel_first，就要对数据维度顺序进行一下调整
        if not K.image_data_format == 'channel_first':
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
        else:
            print('channel_last')

        # 输出训练集和测试集的数量
        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')

        # 后面模型中会使用categorical_crossentropy作为损失函数，这里要对类别标签进行One-hot编码
        train_labels = keras.utils.to_categorical(train_labels, nb_classes)
        test_labels = keras.utils.to_categorical(test_labels, nb_classes)

        # 图像归一化
        train_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels


class Model:
    # 初始化
    def __init__(self):
        self.model = None

    # 建立模型
    def build_model(self, dataset, nb_classes=class_num):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=dataset.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # strides默认等于pool_size
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

    # 训练模型
    def train(self, dataset, batch_size=55, nb_epoch=3, data_augmentation=True):
        self.model.compile(loss='categorical_crossentropy', optimizer='ADAM', metrics=['accuracy'])
        if not data_augmentation:
            self.model.fit(dataset.train_images, dataset.train_labels, batch_size=batch_size, epochs=nb_epoch,
                           shuffle=True)
        else:
            datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, horizontal_flip=True)
            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels, batch_size=batch_size),
                                     epochs=nb_epoch, steps_per_epoch=40)

    # 验证模型
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels)
        print("%s: %.3f%%" % (self.model.metrics_names[1], score[1] * 100))

    def save_model(self, model_path):
        self.model.save(model_path)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.load()
    model = Model()
    model.build_model(dataset, class_num)
    model.train(dataset)
    model.evaluate(dataset)
    model.save_model('./model/face_cnn.model.h5')
