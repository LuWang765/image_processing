import cv2 as cv
import numpy as np


def sobel_demo():
    img = cv.imread(r"F:\project\Python\CV_project_practice\cv_practice\graph\graph_ori.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    grade_x = cv.Scharr(binary, cv.CV_32F, 1, 0)
    grade_y = cv.Scharr(binary, cv.CV_32F, 0, 1)
    cv.imshow("grade_x", grade_x)
    cv.imshow("grade_y", grade_y)

    grade_xy = cv.addWeighted(grade_x, 0.5, grade_y, 0.5, 0)
    cv.imshow("grade_xy", grade_xy)


#礼帽操作
def blackhat_opration(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel=kernel)
    cv.imshow("blackhat", blackhat)


#礼帽操作
def tophat_opration(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel=kernel)
    cv.imshow("tophat", tophat)


#梯度操作
def gradient_opration(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel=kernel)
    cv.imshow("gradient", gradient)


#开操作
def close_opration(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=kernel)
    cv.imshow("close", close)


#开操作
def open_opration(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    open = cv.morphologyEx(img, cv.MORPH_OPEN, kernel=kernel)
    cv.imshow("open", open)

#腐蚀操作
def erode_opration(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    erode = cv.morphologyEx(img, cv.MORPH_ERODE, kernel=kernel)
    cv.imshow("erode", erode)


#膨胀操作
def dilate_operation(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilate = cv.morphologyEx(img, cv.MORPH_DILATE, kernel=kernel)
    cv.imshow("dilate", dilate)


if __name__ == "__main__":
    img = cv.imread(r"F:\project\Python\CV_project_practice\cv_practice\graph\graph_ori.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # cv.imshow("binary", binary)
    #
    # dilate_operation(binary)
    #
    # erode_opration(binary)
    # cv.waitKey(0)
    sobel_demo()
    cv.waitKey(0)
    cv.destroyAllWindows()