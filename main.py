from skimage import io, color, filters, morphology, feature, img_as_ubyte
from skimage.morphology import square
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlim, plot
from itertools import chain
import cv2
import numpy as np



def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        return [px, py]


def main():
    d = {}
    d[0] = "./zdjecia/1.jpg"
    d[1] = "./zdjecia/grr.jpg"
    d[2] = "./zdjecia/fff.jpg"
    d[3] = "./zdjecia/grrr.jpg"
    d[4] = "./zdjecia/44.jpg"
    d[5] = "./zdjecia/beuken-vineer-9x9-2-.jpg"
    d[6] = "./zdjecia/dd2.jpg"
    d[7] = "./zdjecia/33.jpg"
    d[8] = "./zdjecia/rr.jpg"
    d[9] = "./zdjecia/rr2.jpg"
    d[10] = "./zdjecia/tt1.jpg"
    d[11] = "./zdjecia/tt2.jpg"

    for i in range(len(d)):
        img = cv2.imread(d[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret,thresh1 = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)

        imgEnd = cv2.Canny(thresh1,50,150,apertureSize = 3)

        kernel = np.ones((2,2),np.uint8)
        dil = cv2.dilate(imgEnd,kernel,iterations = 1)

        lines = cv2.HoughLines(dil,1,np.pi/180,250)
        if(lines is None or len(lines) < 9):
            imgEnd = cv2.Canny(gray, 50, 150, apertureSize = 3)
            dil = cv2.dilate(imgEnd, kernel, iterations = 1)
            lines = cv2.HoughLines(dil, 1, np.pi/180, 400)
            
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

        poziome = np.copy(lines)
        pionowe = np.copy(lines)
        for ii in reversed(range(len(lines))):
            rho, theta = lines[ii][0]
            if(1.40 < theta < 1.60):
                pionowe = np.delete(pionowe, ii, axis=0)
            else:
                poziome = np.delete(poziome, ii, axis=0)
        
        circles = np.empty((len(poziome) * len(pionowe), 2), dtype=np.float32)

        imgCopy = img.copy()
        nr = 0
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
                circles[nr] = middle_of_circle
                nr += 1
               
        ind = np.lexsort((circles[:,0], circles[:, 1]))
        circles = circles[ind]
        
        imgCopy = img.copy()
        
        for ii in circles:
            r = gray[int(ii[1])-10:int(ii[1])+10, int(ii[0]) - 10:int(ii[0]) + 10 ]
            r2 = r.tolist()
            r2 = list(chain.from_iterable(r2))
            lenr2 = len(r2)
            pcolor = sum(r2) / lenr2
            if pcolor > 220:
                cv2.putText(imgCopy, "W", (int(ii[0]), int(ii[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA )
            elif pcolor < 80:
                cv2.putText(imgCopy, "B", (int(ii[0]), int(ii[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA )
            else:
                cv2.putText(imgCopy, "o", (int(ii[0]), int(ii[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA )


        cv2.imwrite("s{}6.jpg".format(i), imgCopy) #ostateczny obraz 

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()