# 人脸识别

两种方案：CNN、KNN+Facenet

1、运行getdata.py，收集人脸信息，一次一人。代码开头调整名字和图片数量，按q退出，按r开始收集。<br>
2、运行train_cnn或train_knn。<br>
3、运行main_face.py，代码开头调整帧率，需要在代码内部选择cnn或knn（注释和取消注释），默认knn。<br>
注：cnn较慢，knn较快。knn要求图片大小为160。
