import cv2

if __name__ == "__main__":
    im = cv2.imread("IMG_6618.JPG", 0)
    print im.size()
    cv2.namedWindow("trial", 1)
    cv2.imshow("trial", im)