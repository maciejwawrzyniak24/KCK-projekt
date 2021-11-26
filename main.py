from skimage import io, color, filters, morphology, feature
from skimage.morphology import square
from matplotlib import pyplot as plt

def thresholdWhite(table):
    thresh = 0.75
    return table > thresh

def thresholdBlack(table):
    thresh = 0.20
    return table < thresh

def morph(table, func, erosion, dilatation):
    table1 = func(table)
    table1 = morphology.erosion(table1, square(erosion))
    table1 = morphology.dilation(table1, square(dilatation))
    return table1

def main():
    d = {}
    d[0] = "./zdjecia/VVBqkkKdOPceJJAwERLFOxcfQ8Y.jpg"
    d[1] = "./zdjecia/deska-0x400-ffffff.jpg"
    d[2] = "./zdjecia/unnamed.jpg"

    for i in range(len(d)):
        img = io.imread(d[i], as_gray=True)

        img1 = morph(img, thresholdWhite, 3, 5)
        img2 = morph(img, thresholdBlack, 5, 5)

        plt.subplot(2,3,2*i+1)
        plt.imshow(img1, cmap='gray')
        plt.subplot(2,3,2*i+2)
        plt.imshow(img2, cmap='gray')
    plt.show()



if __name__ == '__main__':
    main()