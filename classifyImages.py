import cv2
import numpy as np
import glob, os
from matplotlib import pyplot as plt
from cropImages import crop
import csv


def writeRow(outputFile, featureList):
    wr = csv.writer(outputFile, quoting=csv.QUOTE_ALL)
    wr.writerow(featureList)
    

def getImageBrigthness(img):
    height, width, channels = img.shape
    total = 0
    r=0
    g=0
    b=0
    count = 0
    for y in range(height):
        for x in range(width):
            total += (img[y][x][0] + img[y][x][1] + img[y][x][2])//3
            r += img[y][x][0]
            g += img[y][x][1]
            b += img[y][x][2]
            count = count + 1
    return total//count, r//count, g//count, b//count

def main():
    outPutFileName = "features.csv"
    outPutFile = open(outPutFileName, 'wb')
    fileHeaders = ["id","relPositionZ","positionX","positionY","bestProbabilityRatio","bestProbability"]
    writeRow(outPutFile,fileHeaders)
    baseTemplate = cv2.imread('template.jpeg')
    bestProb = 0
    bestRatio = 0
    bestLayer = 0
    bestBottom_right = (0,0)
    bestTop_left = (0,0)
    bestProbImg = 0
    bestProbFile = "";
    imgDir = "/images/"
    os.chdir(os.getcwd()+imgDir)
    for dirName, subdirList, fileList in os.walk(os.getcwd()):
        if not dirName.endswith("images"):
            print("Sequence: " + str(dirName))
            print(fileList)
            imgcodes = []
            layerCount = 1
            for f in fileList:
                if f.endswith('.jpeg') or f.endswith('.jpg'):
                    layerCount += 1
                    filePath = str(dirName) + "/"
                    print("File:" + dirName + "/" +  str(f))
                    crop(filePath,str(f))
                    fileName = str(dirName) + "/" + "cropped_" + f
                    img = cv2.imread(fileName)
                    os.remove(fileName)
                    #print(str(getImageBrigthness(img)))
                    img2 = img.copy()
                    templateSizingSteps = np.arange(0.5,2,0.5)
                    for step in templateSizingSteps:
                        print("Current resizing step: " + str(step))
                        template = baseTemplate
                        w, h, c = template.shape
                        print("template: " +str(template.shape[:2]))
                        template = cv2.resize(template,(int(h*step), int(w*step)), interpolation = cv2.INTER_CUBIC)      
                        # All the 6 methods for comparison in a list
            #            methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #                        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
                        methods = ['cv2.TM_SQDIFF_NORMED']
                        for meth in methods:
                            img = img2.copy()
                            method = eval(meth)
                            # Apply template Matching
                            res = cv2.matchTemplate(img,template,method)
                            print("res:" + str(res.shape[:2]))
                            print("img:" + str(img.shape[:2]))
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                            #print("max_val" + str(max_val))
                            #print("min_loc" + str(min_loc))
                            #print("max_loc" + str(max_loc))
                            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                                top_left = min_loc
                            else:
                                top_left = max_loc
                            bottom_right = (top_left[0] + int(w*step), top_left[1] + int(h*step))

                            sumProbability = 0
                            sumProbabilityOuter = 1
                            sumProbabilityInner = 0
                            adjustedRes = res
                            resWidth, resHeight = adjustedRes.shape
                            adjustedResHeigth = int(resHeight + h*step)
                            adjustedResWidht = int(resWidth + w*step)
                            adjustedRes = cv2.resize(adjustedRes,( adjustedResHeigth, adjustedResWidht), interpolation = cv2.INTER_CUBIC)

                            print("adjustedRes:" + str(adjustedRes.shape[:2]))
                            for x in range(top_left[0],top_left[0]+int(w*step),1):
                                for y in range(top_left[1],top_left[1] + int(h*step),1):
                                    #if(x < len(res) and y <len(res[0])):
                                    sumProbability += adjustedRes[y][x]
                                        # automatic penalize for being to close to corner
                                    if (x < (top_left[0]+int(w*step)/3) or y < (top_left[1] + int(h*step)/3)):
                                        sumProbabilityOuter += adjustedRes[y][x]
                                    elif  (x > (top_left[0]+int(w*step)/3*2) or y > (top_left[1] + int(h*step)/3*2)):
                                        sumProbabilityOuter += adjustedRes[y][x]
                                    else:
                                        sumProbabilityInner += adjustedRes[y][x]
                                        
                            #print("inner: " + str(sumProbabilityInner))
                            #print("outer: " + str(sumProbabilityOuter))
                            print("ratio: " + str(sumProbabilityInner/sumProbabilityOuter))
                            prob = sumProbability/(w*step*h*step)
                            if (sumProbabilityInner/sumProbabilityOuter) > bestRatio:
                                print("New top")
                                bestRatio = (sumProbabilityInner/sumProbabilityOuter)
                                bestLayer = layerCount
                                bestProbImg = img
                                bestProb = prob
                                bestBottom_right = bottom_right
                                bestTop_left = top_left
                                #cv2.rectangle(img,bestTop_left, bestBottom_right, 255, 1)
                                #plt.subplot(121),plt.imshow(bestProbImg,cmap = 'gray')
                                #plt.title('Best Match: ' + str((sumProbabilityInner/sumProbabilityOuter))), plt.xticks([]), plt.yticks([])
                                #plt.subplot(122),plt.imshow(baseTemplate,cmap = 'gray')
                                #plt.title('Template'), plt.xticks([]), plt.yticks([])
                                #plt.show()
                            print("prob: " + str(prob))
                            print("top_left: " + str(top_left))
                            print("bottom_right: " + str(bottom_right))
            features = [str(dirName),str(bestLayer/layerCount),str((bestBottom_right[0] - bestTop_left[0])/2),\
                        str((bestBottom_right[1] - bestTop_left[1])/2),str(bestRatio),str(bestProb)]
            writeRow(outPutFile, features)
#            cv2.rectangle(bestProbImg,bestTop_left, bestBottom_right, 255, 1)
#            plt.imshow(bestProbImg,cmap = 'gray')
#            plt.title('Best Match('+str(directory) + '): ' + str(bestProb)), plt.xticks([]), plt.yticks([])
#            plt.show()

main()
