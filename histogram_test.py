import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import myutils
import argparse

def plot_demo(src):
    plt.hist(src.ravel(), 256, [0, 256])
    plt.show()

def image_hist(image):
    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        hist = cv.calcHist(image, [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


def equalHist_test(image):  #图像均衡化提高对比度
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow("equalHist_test", dst)


def clahe_test(image):    #局部均衡化提高对比度
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(gray)
    cv.imshow("clahe_test", gray)


def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros((16*16*16, 1), np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgbHist[index, 0] += 1
    return rgbHist


def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    print("巴氏距离: %s." %match1)


def hist2D_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    hist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(hist, interpolation="nearest")
    plt.title("hist2D_demo")
    plt.show()


#def back_projection_demo():  #图像反向投影


def threshold_demo(image):   #全局阈值二值化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    print("ret:%d"%ret)
    cv.imshow("threshold_demo", binary)


def local_threshold(image):   #局部阈值二值化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow("local_threshold", dst)


#def big_image_binary(image):   #超大图像二值化


def pyramid_demo(image):     #金字塔
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_demo"+str(i), dst)
        temp = dst.copy()
    return pyramid_images


def sobel_demo(image):
    grade_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grade_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    grade_x = cv.convertScaleAbs(grade_x)
    grade_y = cv.convertScaleAbs(grade_y)
    cv.imshow("sobel_demo_x", grade_x)
    cv.imshow("sobel_demo_y", grade_y)

    gradexy = cv.addWeighted(grade_x, 0.5, grade_y, 0.5, 0)
    cv.imwrite("mdoel.jpg", gradexy)
    cv.imshow("gradexy", gradexy)

# Canny算法介绍-5步：
# 1、高斯模糊；
# 2、灰度转换；
# 3、计算梯度；
# 4、非最大信号抑制；
# 5、高低阈值输出二值图像
def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  #降低噪声
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    xgrade = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrade = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    # edge_output = cv.Canny(xgrade, ygrade, 50, 150)
    edge_output = cv.Canny(gray, 50, 150)    #对噪声敏感的边缘处理算法
    cv.imshow("edge_demo", edge_output)

    dst = cv.bitwise_and(image, image, mask=edge_output)
    cv.imshow("edge_demo_color", dst)


def line_detection(image):        #霍夫直线检测（手动）
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000 * a)
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detection", image)


def line_detect_possible_demo(image):   #霍夫直线检测（自动）
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    for line in lines:
        print(type(lines))
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detect_possible_demo", image)


# 霍夫圆检测现实考量：
# 1、因为霍夫圆检测对噪声比较敏感，所以先要对图像做中值滤波；
# 2、基于效率考虑，Opencv中实现的霍夫变换圆检测是基于图像梯度的实现，分为两步：
#     1）边缘检测，发现可能的圆心；
#     2）基于第一步的基础上从候选圆心开始计算最佳半径的大小。
def detect_circles_demo(image):     #霍夫圆检测
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)
    #dst = cv.GaussianBlur(image, (1, 1), 0)  # 降低噪声
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)

    cv.imshow("detect_circles_demo", image)


# 轮廓发现介绍（基于拓扑结构）：
#     是基于图像边缘提取的基础寻找对象轮廓的方法，所以边缘提取的阈值选取会影响最终轮廓发现结果。
def contours_demo(image):
    dst = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), -1)

    cv.imshow("contours_demo", image)


# 对象测量：
# 1、弧长和面积（轮廓发现；计算每个轮廓的弧长和面积，像素单位）；
# 2、多边形拟合（获取轮廓的多边形拟合结果）；
# 3、几何矩计算（原点矩，中心炬）。test4
def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   #contours保存所有轮廓信息。
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        moments = cv.moments(contour)
        type(moments)
        cx = moments['m10']/moments['m00']
        cy = moments['m01']/moments['m00']
        cv.circle(dst, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)
        # cv.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
        approx = cv.approxPolyDP(contour, 4, True)   #epsilon一般跟周长相关
        print(approx.shape)
        if approx.shape[0] > 6:
            cv.drawContours(dst, contours, i, (255, 0, 0), 2)
        if approx.shape[0] == 6:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
        if approx.shape[0] == 5:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)
        if approx.shape[0] == 4:
            cv.drawContours(dst, contours, i, (255, 0, 255), 2)
        if approx.shape[0] == 3:
            cv.drawContours(dst, contours, i, (255, 255, 0), 2)
    cv.imshow("measure_object", dst)


#膨胀与腐蚀（最大值滤波）
#图像形态学：
# 1、是图像处理科学的一个单独分支科学；
# 2、灰度与二值图像处理中重要手段；
# 3、是由数学的集合论等相关理论发展起来的。

# 膨胀的作用：
# 1、对象大小增加一个像素（3×3）；
# 2、平滑对象边缘；
# 3、减少或者填充对象之间的距离。

# 腐蚀的作用:
# 1、对象大小减少一个1个像素（3×3）
# 2、平滑对象边缘；
# 3、弱化或者分割图像之间的半岛链接
def erode_demo(image):#腐蚀
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.erode(binary, kernel)
    cv.imshow("erode_demo", dst)


def dilate_demo(image):#膨胀
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.dilate(binary, kernel)
    cv.imshow("dilate_demo", dst)


def oringin_erode_demo(image):#原图腐蚀
    print(image.shape)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.erode(image, kernel)
    cv.imshow("oringin_erode_demo", dst)


def origin_dilate_demo(image):#原图膨胀
    print(image.shape)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.dilate(image, kernel)
    cv.imshow("origin_dilate_demo", dst)

    
# 开操作（open，去除噪点）
# 1、图像形态学的重要操作之一，基于膨胀与腐蚀操作组合形成的；
# 2、主要是应用在二值图像分析中，灰度图像亦可；
# 3、开操作=腐蚀+膨胀，输入图像+结构元素。

#闭操作（close， 填充小区域）
# 1、图像形态学的重要操作之一，基于膨胀与腐蚀操作组合形成的；
# 2、主要应用在二值图像分析中，灰度图像亦可；
# 3、闭操作=膨胀+腐蚀，输入图像+结构元素。

# 开闭操作作用：
# 1、去除晓得干扰快-开操作；
# 2、填充闭合区域-闭操作；
# 3、水平或者垂直线提取。
def open_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("open_demo", dst)


def close_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow("close_demo", dst)


#顶帽：原图像与开操作之间的差值图像
#黑帽：闭操作图像与原图像之间的差值图像
# 形态学梯度
# 基本梯度是用膨胀后的图像减去腐蚀后的图像得到差值图像，成为基本梯度；
# 内部梯度是用原图减去腐蚀之后的图像得到差值图像，成为图像的内部梯度；
# 外部梯度是图像膨胀之后的图像减去原来的图像的差值图像，称为图像的外部梯度。
def top_hat_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    dst = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)
    cv.imshow("close_demo", dst)


# 分水岭算法：距离变换。
# 输入图像->灰度->二值化->距离变换->寻找种子->生成Marker->分水岭变换->输出图像->End
def wartershed_demo():#coins.jpg
    print(src.shape)
    blurred = cv.pyrMeanShiftFiltering(src, 10, 100)

    # gray\binary image
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # mophology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    nb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(nb, kernel, iterations=3)
    cv.imshow("mophology", sure_bg)

    #distance transform
    dist = cv.distanceTransform(nb, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow("distance", dist_output*50)

    ret, surface = cv.threshold(dist, dist.max()*0.6, 255, cv.THRESH_BINARY)
    cv.imshow("surface", surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)

    #wartershed transform
    markers = markers +1
    markers[unknown==255] = 0
    markers = cv.watershed(src, markers=markers)
    src[markers==-1] = (0, 0, 255)
    cv.imshow("result", src)


#人脸数据
def face_detect_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier(r"F:\project\Python\house predict\haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray, 1.1, 2)     #后两个参数调整可提高人脸检测成功率
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("result", image)



print("--------------Hellp world!---------------")
src = cv.imread(r"F:\project\Python\house predict\test.jpg"); #type: cv
src = myutils.resize(src, width=1600)
cv.namedWindow("result", cv.WINDOW_AUTOSIZE)

# capture = cv.VideoCapture(0)
# while True:
#     ret, frame = capture.read()
#     frame = cv.flip(frame, 1)
#     break;
#     # face_detect_demo(frame)
#     # c = cv.waitKey(10)
#     # if 27 == c:
#     #     break
# #face_detect_demo(frame)

cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)

cv.imshow("input image1", src)
# face_detect_demo(src)
sobel_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()