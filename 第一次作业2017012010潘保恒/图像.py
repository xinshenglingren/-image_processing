# 目标： 基于PIL 实现图像的读取、保存、可视化、裁剪、旋转
# 0.从PIL中导入模块Image：Python Imaging Library (PIL)
from PIL import Image
from PIL import ImageEnhance

# 1-1.图像数据的读取
img = Image.open('image2.png')

# 1-2.Save the image with a different name
img.save('new_image2.png')

# 1-3.可视化、图像数据信息的获取
img.show()
print('1--原始图像信息：', img.info)

#2-1.Convert the image to grayscale、可视化、信息获取
gray_image = img.convert("L")
gray_image.show()
print('2--转化后的图像信息：', gray_image.info)

# 3. Create a dimension tuple
dim = (100, 100, 400, 400)
crop_img = img.crop(dim)
crop_img.show()

# 4.Rotate the image by 90 degress anti-clockwise
# 4-1.方式1.
rotated_img1 = img.rotate(90)
rotated_img1.show()

# 4-2.方式2.
rotated_img2 = img.rotate(90,resample=Image.BICUBIC,expand=1)
rotated_img2.show()


# 5.灰度图像的亮度增强,注意函数的增强系数2,0.5
enchancer = ImageEnhance.Brightness(gray_image)
bright_img = enchancer.enhance(2)
bright_img.show()

bright_img = enchancer.enhance(0.5)
bright_img.show()


# 7.Resize the image,注意不同的处理方式
resize_img1 = img.resize((300, 300))
resize_img1.show()

resize_img2 = img.resize((300, 300),resample = Image.BICUBIC)
resize_img2.show()
print('3--image size after resizing：', resize_img2.size)