import cv2
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
base_path = './data/'  # 保存路径

path = base_path + 'yty'  # 当前姓名
fps = 20  # FPS
image_num = 200  # 照片数量
image_size = 160  # 照片大小


def GetData_face():
    classifier = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt2.xml')  # Haar分类器
    if not os.path.isdir(path):
        os.makedirs(path)

    cv2.namedWindow('face', cv2.WINDOW_NORMAL)

    # 摄像头
    cap = cv2.VideoCapture(0)

    num = -1
    time0 = time.time()
    while cap.isOpened():
        if time.time() - time0 < 1 / fps:
            continue
        time0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(64, 64))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                x0 = int(x + w / 8)
                x1 = int(x - w / 8) + w
                cv2.rectangle(frame, (x0 - 5, y - 5), (x1 + 5, y + h + 5), (255, 0, 0), 2)
                if num >= 0 and num < image_num:
                    img_name = '%s/%d.jpg' % (path, num)
                    image = cv2.resize(frame[y:y + h, x0:x1], (image_size, image_size))
                    cv2.imwrite(img_name, image)
                    print(num)
                    num += 1

        cv2.imshow('face', frame)
        key = cv2.waitKey(10)

        if key & 0xFF == ord('r'):
            num = 0
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    GetData_face()
