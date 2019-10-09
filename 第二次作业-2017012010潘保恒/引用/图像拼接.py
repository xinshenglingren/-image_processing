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
    refer_X = np.delete(refer_X, 2, axis=1)
    W = np.linalg.inv(input_X.T * input_X) * input_X.T * refer_X
    W = np.insert(W, 2, np.mat([0,0,1]), axis=1)
    return W

#获得灰度图片的大小及灰度矩阵
def get_picture_mess(img):
    ord_img = Image.open(img)
    gray_img = ord_img.convert('L')
    img_arr = np.array(gray_img)
    img_shape = img_arr.shape
    return img_arr, img_shape

#构造拼接后图像的画布
def create_final_img(refer_shape, input_shape, W):
    input_left_up = np.mat([0, 0, 1]) * W
    for i in range(input_left_up.shape[1]):
        input_left_up[0, i] = int(round(input_left_up[0, i]))
    input_left_down = np.mat([input_shape[0], 0, 1]) * W
    for i in range(input_left_down.shape[1]):
        input_left_down[0, i] = int(round(input_left_down[0, i]))
    input_right_up = np.mat([0, input_shape[1], 1]) * W
    for i in range(input_right_up.shape[1]):
        input_right_up[0, i] = int(round(input_right_up[0, i]))
    input_right_down = np.mat([input_shape[0], input_shape[1], 1])
    for i in range(input_right_down.shape[1]):
        input_right_down[0, i] = int(round(input_right_down[0, i]))
    row = [0, refer_shape[0], input_left_up[0, 0], input_right_down[0, 0], input_left_down[0, 0], input_right_down[0, 0]]
    row_min = np.min(row)
    row_max = np.max(row)
    row_num = row_max - row_min
    col = [0, refer_shape[1], input_left_up[0, 1], input_right_down[0, 1], input_left_down[0, 1], input_right_down[0, 1]]
    col_min = np.min(col)
    col_max = np.max(col)
    col_num = col_max - col_min
    flag = np.zeros(int(row_num) * int(col_num)).reshape(int(row_num), int(col_num))

    return flag, -row_min, -col_min

#近邻法灰度插值图像生成
def create_gray_picture(flag, refer_arr, refer_shape, input_arr, input_shape, W, row_poor, col_poor):
    flag_shape  = flag.shape

    for row in range(flag_shape[0]):
        for col in range(flag_shape[1]):
            ord_point = np.mat([row-row_poor, col-col_poor, 1])
            ord_point[0, 0] = int(round(ord_point[0, 0]))
            ord_point[0, 1] = int(round(ord_point[0, 1]))
            if int(ord_point[0, 0]) >= 0 and int(ord_point[0, 0]) <= refer_shape[0] and int(ord_point[0, 1]) >=0 and int(ord_point[0, 1]) <= refer_shape[1]:
                flag[row, col] = refer_arr[int(ord_point[0, 0]), int(ord_point[0, 1])]
            else:
                ord_point = np.mat([row - row_poor, col - col_poor, 1]) * np.linalg.inv(W)
                ord_point[0, 0] = int(round(ord_point[0, 0]))
                ord_point[0, 1] = int(round(ord_point[0, 1]))
                if int(ord_point[0, 0]) >= 0 and int(ord_point[0, 0]) <= input_shape[0] and int(ord_point[0, 1]) >=0 and int(ord_point[0, 1]) <= input_shape[1]:
                    flag[row, col] = input_arr[int(ord_point[0, 0]), int(ord_point[0, 1])]
    return flag


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

refer_arr, refer_shape = get_picture_mess('klcc_a.png')
input_arr, input_shape = get_picture_mess('klcc_b.png')

W = calcute_W(refer_X, input_X)

flag, row_poor, col_poor = create_final_img(refer_shape, input_shape, W)

flag_arr = create_gray_picture(flag, refer_arr, refer_shape, input_arr, input_shape, W, row_poor, col_poor)

# 近邻法插值法得到截取后的灰度图片
img3 = Image.fromarray(flag_arr)
cv2.imwrite('拼接.jpg', flag_arr)
img3.show()
