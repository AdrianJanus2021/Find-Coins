import numpy as np
import cv2 as cv

#%%
#import image, blur it, and convert to grayscale colorspace
imgOrignal = cv.imread('tray.jpg', cv.IMREAD_COLOR)
imgBlur = cv.medianBlur(imgOrignal,3)  
imgGray = cv.cvtColor(imgBlur,cv.COLOR_BGR2GRAY)
#%%
#get binary lvl photo and contours
ret, thresh = cv.threshold(imgGray, 100, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#%%
#find contour of the TRAY - it is contour with the biggest area - using for loop and if condition
imax=0
areamax=0
for i in range(len(contours)):
    temp = contours[i]
    area = cv.contourArea(temp)
    if area > areamax:
        imax=i
        areamax=area
tray = contours[imax]
#where imax is the number of the contour with the biggest area
area = cv.contourArea(tray)
cv.drawContours(imgOrignal, [tray], 0, (0,255,0), 3)
#%%
#find circles
circles = cv.HoughCircles(imgGray,method=cv.HOUGH_GRADIENT,dp=1,minDist=20, param1=40,param2=40,minRadius=20,maxRadius=50)
circles = np.uint16(np.around(circles))
#draw circles
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(imgOrignal,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(imgOrignal,(i[0],i[1]),2,(0,0,255),3)
#%%
#calculate coins
BigCoinInTray=0
SmallCoinInTray=0
BigCoinOutTray=0
SmallCoinOutTray=0
for i in circles[0,:]:
    if cv.pointPolygonTest(tray,(i[0],i[1]),False)>-1:
        if i[2] > 31:
            BigCoinInTray=BigCoinInTray+1
        else:
            SmallCoinInTray=SmallCoinInTray+1
    else:
        if i[2] > 31:
            BigCoinOutTray=BigCoinOutTray+1
        else:
            SmallCoinOutTray=SmallCoinOutTray+1

imgOrignal = cv.putText(imgOrignal,"BigCoinInTray = "+str(BigCoinInTray), (50,50),fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)
imgOrignal = cv.putText(imgOrignal,"BigCoinOutTray = "+str(BigCoinOutTray), (50,100),fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)
imgOrignal = cv.putText(imgOrignal,"SmallCoinInTray = "+str(SmallCoinInTray), (50,150),fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)
imgOrignal = cv.putText(imgOrignal,"SmallCoinOutTray = "+str(SmallCoinOutTray), (50,200),fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)
imgOrignal = cv.putText(imgOrignal,"Area = "+str(areamax), (50,250),fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)

#%%
#display results
imgRes = cv.resize(imgOrignal, (0,0), fx=0.8, fy=0.8) 
cv.imshow('COINS',imgRes)
cv.waitKey(0)
cv.destroyAllWindows()