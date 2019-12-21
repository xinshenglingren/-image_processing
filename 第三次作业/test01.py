from PIL import Image
from PIL import ImageEnhance 
import numpy as np
def depoint(img):
    """传入二值化后的图片进行降噪"""
    pixdata = img.load()
    w,h = img.size
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] == 0:#上
                count = count + 1
            if pixdata[x,y+1] == 0:#下
                count = count + 1
            if pixdata[x-1,y] == 0:#左
                count = count + 1
            if pixdata[x+1,y] == 0:#右
                count = count + 1
            if pixdata[x-1,y-1] == 0:#左上
                count = count + 1
            if pixdata[x-1,y+1] == 0:#左下
                count = count + 1
            if pixdata[x+1,y-1] == 0:#右上
                count = count + 1
            if pixdata[x+1,y+1] == 0:#右下
                count = count + 1
            if count < 4:
                pixdata[x,y] = 1
    return img


def tow_value(imgry):
    max = 0
    gray = 0
    data =[]
    for i in imgry.getcolors():
        if i[0]>max:
            max = i[0]
            gray = i[1]
        if i[0] < 42:
            data.append(i[1])
    data.append(gray)
    def get_bin_table(data):
        table = []
        for i in range(256):
            if i in data:
                table.append(0)
            else:
                table.append(1)

        return table

    table = get_bin_table(data)
    out = imgry.point(table, '1')
    return out


#有效点检测
def effective(image,x,y):
    pixdata = image.load()
    count = 0
    if pixdata[x,y-1] == 0:#上
        count = count + 1
    if pixdata[x,y+1] == 0:#下
        count = count + 1
    if pixdata[x-1,y] == 0:#左
        count = count + 1
    if pixdata[x+1,y] == 0:#右
        count = count + 1
    if pixdata[x-1,y-1] == 0:#左上
        count = count + 1
    if pixdata[x-1,y+1] == 0:#左下
        count = count + 1
    if pixdata[x+1,y-1] == 0:#右上
        count = count + 1
    if pixdata[x+1,y+1] == 0:#右下
        count = count + 1
    if count >= 6:
        return 1
    else:
        return 0


def pcf(image,index):
    for i in range(5):
        for x in range(index[i][1],(index[i][1]+index[i][3])):
            image.putpixel((x,index[i][0]),(0,0,0,0))
            image.putpixel((x,index[i][0]+index[i][2]),(0,0,0,0))
        for y in range(index[i][0],(index[i][0]+index[i][2])):
            image.putpixel((index[i][1],y),(0,0,0,0))
            image.putpixel((index[i][1]+index[i][3],y),(0,0,0,0))
    return image

def effective_index(image):
    w,h = image.size
    # 列的有效值提取
    index = []
    for x in range(3,w-3):
        for y in range(3,h-3):          
            if effective(image,x,y):
                index.append(x)
                break
    x_index  = extract(index,1)
    if x_index:
        #行的有效值提取
        y_index = []
        for i in range(0,10,2):
            index = []
            cropped_image = image.crop((x_index[i], 0, x_index[i+1], h))
            pixdata = cropped_image.load()
            w,h = cropped_image.size
            for y in range(3,h-3):
                for x in range(3,w-3):
                    if effective(cropped_image,x,y):
                        index.append(y)
                        break
            eff_index = extract(index,0)
            if eff_index:
                y_index.append(eff_index[0])
                y_index.append(eff_index[1])
        if y_index.__len__() == 10:
            index = [[],[],[],[],[]]
            j = 0
            for i in range(0,10,2):
                index[j].append(y_index[i]-2)
                index[j].append(x_index[i]-2)
                index[j].append(y_index[i+1] - y_index[i]+4)
                index[j].append(x_index[i+1] - x_index[i]+4)
                j+=1
            return index
        else:
            return 0
    else:
        return 0

#index提取有效值(coord为坐标，若对X轴提取输入1，若对Y轴提取输入0)
def extract(ls,coord): 
    result= [] #存储有效值
    judge = 0 #判断当前值是否为连续值得起始值(0为起始)
    jdg = 0 #判断e_num是否被赋值（为1则被赋值）
    s_num = 0  #记录当前连续的起始值
    e_num = 0  #记录当前连续的终止值
    ls.append(200) #此200为无效值，用来便于统计有效值
    for i in range(ls.__len__()-1):
        #清除孤立值
        if i == 0 and ls[i+1] - ls[i] > 1:
            continue
        if ls[i] - ls[i-1] > 1 and ls[i+1] -ls[i] > 1:
            continue
            
        if ls[i+1] - ls[i] == 1 and judge == 0:
            s_num = ls[i]
            judge = 1
        elif ls[i+1] - ls[i] == 1 and judge == 1:
            continue
        else:
            e_num = ls[i]
            judge = 0
            jdg = 1
        
        if jdg == 1 and ls.index(e_num) - ls.index(s_num) > 7: #长度大于8的为有效字符长度
            result.append(s_num)
            result.append(e_num)
            jdg = 0
        else:
            jdg = 0
    if coord:
        if result.__len__() == 10:
            return result
        else:
            return 0
    else:
        if result.__len__() == 2:
            return result
        else:
            return 0

n_im= Image.new("RGB", (600, 750)) #创建一张大图
col = 1 #检测当前图像在第几列
row = 1 #检测当前图像在第几行
num = 0 #检测加载的图片数目
invalid = []#有效图片名称
for k in range(115):
    orig_image = Image.open('有用2/%d.jpg'%k)
    imgry = orig_image.convert('L')  # 转化为灰度图
    deal_image = tow_value(imgry)
    #降噪6次
    for i in range(6):
        deal_image = depoint(deal_image)
    index = effective_index(deal_image)
    if index:
        deal_orig_image = pcf(orig_image,index)
        for x in range((col-1) * 150,col * 150):
            for y in range((row-1)* 30,row * 30):
                n_im.putpixel((x,y),(deal_orig_image.getpixel((x-((col-1) * 150),y-((row-1) * 30)))))
        invalid.append('%d.jpg'%k)
        num += 1
        col += 1
        if num == 100:
            break
        if num % 4 == 0:
            col = 1
            row += 1

n_im.save('25-4.jpg')

import csv
row = []
for k in invalid:
    index_data = []
    orig_image = Image.open('有用2/%s'%k)
    imgry = orig_image.convert('L')  # 转化为灰度图
    deal_image = tow_value(imgry)
    #降噪6次
    for i in range(6):
        deal_image = depoint(deal_image)
    index = effective_index(deal_image)
    for i in range(5):
        for j in range(4):
            index_data.append(index[i][j])
    data = tuple(['%s'%k] + index_data)
    row.append(data)
with open('25-4.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(row)