import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1-1.图像数据的读取
img = Image.open('人脸图像.jpg')
# 1-2.Save the image with a different name
img.save('new_人脸图像.jpg')
img.show()
#1-3实现人机交互获取输入坐标
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
img = cv2.imread(img)
cv2.namedWindow('image')
data = list()
loc = cv2.setMouseCallback('image', on_EVENT_LBUTTONDOWN)
cv2.imshow('image', img)
if cv2.waitKey(0) == ord('s'):
    cv2.destroyAllWindows()
    
#1-4输出图像到输入图像的几何变换模型的估计
getPos = [
    [119,225],
    [114,286],
    [175,260]
]  
targetPos = [
    [25,24],
    [25,76],
    [75,50]
]  
#1-5齐次化函数
def toHomogeneous(data, axis=1):
    data = np.insert(data, len(data[0]), 1, axis=axis)      #默认插入一列1
    return data

#1-6获得仿射变换矩阵
def getTrans(oriPos, targetPos):
    oriPos = np.mat(oriPos)
    targetPos = np.mat(targetPos)
    trans = np.linalg.inv(oriPos.T*oriPos)*oriPos.T*targetPos
    return trans
getPos = toHomogeneous(getPos)
targetPos = toHomogeneous(targetPos)
trans = getTrans(getPos, targetPos)

def get_xycorner(img):
    xlen, ylen = img.shape[0]-1, img.shape[1]-1      
    xycorner = np.array([
        [0,0,1],
        [0,ylen,1],
        [xlen,0,1],
        [xlen,ylen,1]
    ])      #四个顶点的齐次坐标
    return xycorner
xycorner = get_xycorner(img)

def transform(vectors, transMat):
    return (vectors*transMat)
uvcorner = transform(xycorner, trans)

def get_inv_trans(trans):
    inv_trans = np.linalg.inv(trans)     
    return inv_trans
inv_trans = get_inv_trans(trans)
new_array = np.zeros((math.floor(len_0),math.floor(len_1)))
#获得初始图片的形状以及灰度值
def get_ordPicture_mess(ord_picture):
    ord_img = Image.open(ord_picture)
    gray_img = ord_img.convert('L')
    img_arr = np.array(gray_img)
    img_shape = img_arr.shape
    return  img_arr, img_shape
#最近邻法灰度插值
def create_gray_img_by_neighbor(img_arr, W):
    gray_img_arr = np.zeros(101 * 101).reshape(101, 101)
    for row in range(101):
        for col in range(101):
            point = np.mat([row, col, 1])
            ord_point = point * np.linalg.inv(W)
            gray_img_arr[row, col] = img_arr[int(round(ord_point[0,0])), int(round(ord_point[0,1]))]
    return gray_img_arr
#获得输入灰度图片的灰度值矩阵及形状
img_arr, img_shape = get_ordPicture_mess(ord_picture)
#获得近邻法所得灰度图片矩阵
gray1_img_arr = create_gray_img_by_neighbor(img_arr, W)
Img1 = Image.fromarray(gray1_img_arr)
cv2.imwrite('最近邻法灰度图片.jpg',gray1_img_arr)
img1.show()

#双线性灰度插值法
def create_gray_img_by_bilinear(img_arr, W):
    gray_img_arr = np.zeros(101 * 101).reshape((101, 101))
    for row in range(101):
        for col in range(101):
            point = np.mat([row, col, 1])
            ord_point = point * np.linalg.inv(W)
            x0 = ord_point[0, 0]
            y0 = ord_point[0 ,1]
            x = math.floor(ord_point[0, 0])
            x1 = math.ceil(ord_point[0 ,0])
            y = math.floor(ord_point[0 ,1])
            y1 = math.ceil(ord_point[0, 1])
            a = (x0 - x)/(x1 - x)
            b = (y0 - y)/(y1 - y)
            gray = (1-a)*(1-b)*img_arr[x, y] + a*(1-b)*img_arr[x1, y] + b*(1-a)*img_arr[x, y1] + a*b*img_arr[x1, y1]
            gray_img_arr[row, col] = gray
    return gray_img_arr
#获得输入灰度图片的灰度值矩阵及形状
img_arr, img_shape = get_ordPicture_mess(ord_picture)
#双线性灰度插值法得到的灰度图片矩阵
gray2_img_arr = create_gray_img_by_bilinear(img_arr, W)
Img2 = Image.fromarray(gray2_img_arr)
cv2.imwrite('双线性插值法灰度图片.jpg',gray2_img_arr)
img2.show()

#双三线性插值法