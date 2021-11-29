from skimage import io, color, filters, morphology, feature
from skimage.morphology import square
from matplotlib import pyplot as plt
import cv2
import numpy as np

def main():
    d = {}
    d[0] = "./zdjecia/1.jpg"
    d[1] = "./zdjecia/2.jpg"

    for i in range(len(d)):
        img = cv2.imread(d[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("s{}.jpg".format(i), gray)

        ret,thresh1 = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
        cv2.imwrite("s{}1.jpg".format(i), thresh1)

        imgEnd = cv2.Canny(thresh1,90,150,apertureSize = 3)
        cv2.imwrite("s{}2.jpg".format(i), imgEnd)

        kernel = np.ones((2,2),np.uint8)
        dil = cv2.dilate(imgEnd,kernel,iterations = 1)
        cv2.imwrite("s{}3.jpg".format(i), dil)

        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        cv2.imwrite("s{}35.jpg".format(i), edges)
        lines = cv2.HoughLines(dil,1,np.pi/180,250)

        imgCopy = img.copy()
        j = 0
        for line in lines:
            j+=1
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            #print(a)
            #print(b)
            #print()
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b)) 
            y1 = int(y0 + 2000*(a)) 
            x2 = int(x0 - 2000*(-b)) 
            y2 = int(y0 - 2000*(a)) 
            cv2.line(imgCopy,(x1,y1),(x2,y2),(0,0,255),2)
        
        print("obraz {}, linii {}".format(i,j))

        cv2.imwrite("s{}4.jpg".format(i), imgCopy) #ostateczny obraz
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()