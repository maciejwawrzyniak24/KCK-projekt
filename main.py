from skimage import io, color, filters, morphology, feature, img_as_ubyte
from skimage.morphology import square
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlim, plot
import cv2
import numpy as np



def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        return [px, py]



def main():
    d = {}
    d[0] = "./zdjecia/1.jpg"
    d[1] = "./zdjecia/2.jpg"
    d[2] = "./zdjecia/deska-0x400-ffffff.jpg"
    d[3] = "./zdjecia/262107721_597105391558839_4796743744082034723_n.jpg"
    d[4] = "./zdjecia/dd.jpg"
    d[5] = "./zdjecia/dd2.jpg"

    for i in range(len(d)):
        img = cv2.imread(d[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite("s{}.jpg".format(i), gray)

        #gray = filters.rank.median(gray, np.ones([3,3]))

        ret,thresh1 = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
        #cv2.imwrite("a{}1.jpg".format(i), thresh1)

        imgEnd = cv2.Canny(thresh1,50,150,apertureSize = 3)
        #cv2.imwrite("s{}2.jpg".format(i), imgEnd)

        kernel = np.ones((2,2),np.uint8)
        dil = cv2.dilate(imgEnd,kernel,iterations = 1)
        #cv2.imwrite("d{}3.jpg".format(i), dil)

        #edges = cv2.Canny(gray,50,150,apertureSize = 3)
        #cv2.imwrite("s{}35.jpg".format(i), edges)

        lines = cv2.HoughLines(dil,1,np.pi/180,250)
        #print(len(lines))

        if(len(lines) < 20):
            imgEnd = cv2.Canny(gray, 50, 150, apertureSize = 3)
            dil = cv2.dilate(imgEnd, kernel, iterations = 1)
            lines = cv2.HoughLines(dil, 1, np.pi/180, 200)

        imgCopy = img.copy()

        deleted = []
        for deli in range(len(lines)):
            for delj in range(deli + 1, len(lines)):
                rho1, theta1, = lines[deli][0]
                rho2, theta2, = lines[delj][0]
                if(abs(rho1 - rho2) < 15 and abs(theta1 - theta2) < 0.05):
                    deleted.append(delj)
        deleted = list(set(deleted))
        deleted.sort(reverse=True)
        for deli in deleted:
            lines = np.delete(lines,deli, axis=0)

        j = 0
        for line in lines:
            j+=1
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))
            cv2.line(imgCopy,(x1,y1),(x2,y2),(0,0,255),2)

        #print("obraz {}, linii {}".format(i,j))
        #cv2.imwrite("s{}4.jpg".format(i), imgCopy) #ostateczny obraz

        poziome = np.empty((int(len(lines)/2),1,2), np.float32)
        pionowe = np.empty((int(len(lines)/2),1,2), np.float32)
        j = 0
        k = 0
        for i in range(len(lines)):
            rho, theta = lines[i][0]
            if(1.40 < theta < 1.60):
                poziome[j] = lines[i]
                j += 1
            else:
                pionowe[k] = lines[i]
                k += 1
        
        imgCopy = img.copy()
        for line1 in poziome:
            rho1, theta1 = line1[0]
            a1 = np.cos(theta1)
            b1 = np.sin(theta1)
            x0 = a1*rho1
            y0 = b1*rho1
            x1 = int(x0 + 2000*(-b1))
            y1 = int(y0 + 2000*(a1))
            x2 = int(x0 - 2000*(-b1))
            y2 = int(y0 - 2000*(a1))
            for line2 in pionowe:
                rho2, theta2 = line2[0]
                a2 = np.cos(theta2)
                b2 = np.sin(theta2)
                x3 = a2*rho2
                y3 = b2*rho2
                x4 = int(x3 + 2000*(-b2))
                y4 = int(y3 + 2000*(a2))
                x5 = int(x3 - 2000*(-b2))
                y5 = int(y3 - 2000*(a2))
                middle_of_circle = findIntersection(x1,y1,x2,y2,x4,y4,x5,y5)
                cv2.circle(imgCopy, (int(middle_of_circle[0]), int(middle_of_circle[1])), radius=2, color=(255,0,0), thickness=-1)
               
        cv2.imwrite("s{}5.jpg".format(i), imgCopy) #ostateczny obraz 


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()