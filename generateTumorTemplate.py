import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

def plot(img):
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.show()

    
def generateTemplate(desiredSize):
    imgcodes = range(1,18)
    finalTemplate = np.zeros((desiredSize, desiredSize, 3), np.uint8)
    for imgcode in imgcodes:
        fileName = str(imgcode) + '.PNG'
        if os.path.isfile(os.path.join('templates',fileName)):
            template = cv2.imread(os.path.join('templates',fileName))
            height, width = template.shape[:2]
            #print("Before: " + str(template.shape[:2]))
            template = cv2.resize(template,(desiredSize, desiredSize), interpolation = cv2.INTER_CUBIC)
            #print("After: " + str(template.shape[:2]))
            #plot(template)
            for y in range(desiredSize):
                for x in range(desiredSize):
                    finalTemplate[y][x][0] += template[y][x][0]//len(imgcodes)
                    finalTemplate[y][x][1] += template[y][x][1]//len(imgcodes)
                    finalTemplate[y][x][2] += template[y][x][2]//len(imgcodes)

    #plot(finalTemplate)
    fileName = 'template.jpeg'
    cv2.imwrite(fileName,finalTemplate)

def main():
    generateTemplate(70)

main()
