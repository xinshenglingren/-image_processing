import cv2
import numpy as np
from PIL import Image

#实现人机交互获取输入坐标
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        'x, y = {}, {}'.format(x, y)
        data.append((y,x))
        # print('x, y = {}, {}'.format(y, x))
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
        1.0, (0, 0, 0), thickness=1)
        cv2.imshow('image', img)

#将对比坐标变成矩阵
def create_X(data):
    X = []
    for point in data:
        point = list(point)
        point.append(1)
        X.append(point)
    return np.mat(X)

#获得变换矩阵（前向）
def calcute_W(refer_X, input_X):
    input_X = np.delete(input_X, 2, axis=1)
    W = np.linalg.inv(refer_X.T * refer_X) * refer_X.T * input_X
    W = np.insert(W, 2, np.mat([0,0,1]), axis=1)
    return W

#获得灰度图片的大小及灰度矩阵
def get_picture_mess(img):
    ord_img = Image.open(img)
    gray_img = ord_img.convert('L')
    img_arr = np.array(gray_img)
    img_shape = img_arr.shape
    return img_arr, img_shape

#近邻法灰度插值图像生成
def create_gray_img(refer_arr, refer_shape, input_arr, input_shape, W):
    gray_img1 = np.ones(refer_shape[0] * refer_shape[1]).reshape(refer_shape[0], refer_shape[1])
    gray_img2 = np.ones(refer_shape[0] * refer_shape[1]).reshape(refer_shape[0], refer_shape[1])

    for row in range(refer_shape[0]):
        for col in range(refer_shape[1]):
            point = np.mat([row, col, 1]) * W
            if int(round(point[0, 0])) >= 0 and int(round(point[0, 0])) < input_shape[0] and int(round(point[0, 1])) >= 0 and int(round(point[0, 1])) < input_shape[1]:
                gray_img1[row, col] = refer_arr[row, col]

    for row in range(refer_shape[0]):
        for col in range(refer_shape[1]):
            point = np.mat([row, col, 1]) * W
            if int(round(point[0, 0])) >= 0 and int(round(point[0, 0])) < input_shape[0] and int(round(point[0, 1])) >= 0 and int(round(point[0, 1])) < input_shape[1]:
                gray_img2[row, col] = input_arr[int(round(point[0, 0])), int(round(point[0, 1]))]

    return gray_img1, gray_img2



img = cv2.imread('klcc_a.png')
data = list()
cv2.namedWindow('image')
loc = cv2.setMouseCallback('image', on_EVENT_LBUTTONDOWN)
cv2.imshow('image', img)
if cv2.waitKey(0) == ord('s'):
    refer_data = data
    cv2.destroyAllWindows()

img = cv2.imread('klcc_b.png')
data = list()
cv2.namedWindow('image')
loc = cv2.setMouseCallback('image', on_EVENT_LBUTTONDOWN)
cv2.imshow('image', img)
if cv2.waitKey(0) == ord('s'):
    input_data = data
    cv2.destroyAllWindows()

#参考图片的坐标矩阵
refer_X = create_X(refer_data)
#输入图片的坐标矩阵
input_X = create_X(input_data)

#得到参考图片与输入图片的灰度矩阵的信息
refer_arr, refer_shape = get_picture_mess('klcc_a.png')
input_arr, input_shape = get_picture_mess('klcc_b.png')

#计算参数矩阵
W = calcute_W(refer_X, input_X)

#生成配准后的灰度图片矩阵
gray_img1, gray_img2 = create_gray_img(refer_arr, refer_shape, input_arr, input_shape, W)

#保存配准后的灰度图片
cv2.imwrite('配准1.jpg', gray_img1)
cv2.imwrite('配准2.jpg', gray_img2)

#将灰度矩阵转换为图片
img1 = Image.fromarray(gray_img1)
img2 = Image.fromarray(gray_img2)
# img3.save('PinJie.jpg')
img1.show()
img2.show()
