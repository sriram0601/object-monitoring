import sys
import cv2 as cv
import numpy as np


def image(fileName,save=False):

    img = cv.imread(cv.samples.findFile(fileName))
    if img is None:
        sys.exit('File is empty')
    
    cv.imshow('Display Window',img)
    
    if save==True:
        fname = input('Enter the file name: ')
        cv.imwrite(fname,img)
        
    if cv.waitKey(0) & 0xFF==ord('q'):
        cv.destroyAllWindows()

def drawImage(fileName,shape):
    img = cv.imread(cv.samples.findFile(fileName))    
    if shape.lower() == 'line':
        cord_start_x = int(input('Enter the starting co-ordinate X : '))
        cord_start_y = int(input('Enter the starting co-ordinate Y : '))
        cord_end_X = int(input('Enter the ending co-ordinate X : '))
        cord_end_Y = int(input('Enter the ending co-ordinate Y : '))
        thick = int(input('Enter the thickness of the line: '))

        img = cv.line(img, (cord_start_x,cord_start_y), (cord_end_X,cord_end_Y), (0,255,0), thick)

        cv.imshow('Shapes on Image',img)

        cv.waitKey(0)
        cv.destroyAllWindows()
    
    elif(shape.lower() == 'rectangle'):
        h,w,c = img.shape
        print(f'Height : {h}\tWeight : {w}\tChannel : {c}\n')
        start_point_X = int(input('Enter the starting point X : '))
        start_point_Y = int(input('Enter the starting point Y : '))
        ending_point_X = int(input('Enter the ending point X : '))
        ending_point_Y = int(input('Enter the ending point Y : '))
        thick = int(input('Enter the thickness of the line : '))

        img = cv.rectangle(img, (start_point_X,start_point_Y), (ending_point_X,ending_point_Y), (255,0,255), thick)

        cv.imshow('Shapes in image',img)

        cv.waitKey(0)
        cv.destroyAllWindows()




def findObject(fileName):
    while True:    
        img = cv.imread(fileName)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        l_h = cv.getTrackbarPos('Lower_Hue','Color-Tracker')
        l_s = cv.getTrackbarPos('Lower_Sat','Color-Tracker')
        l_v = cv.getTrackbarPos('Lower_Val','Color-Tracker')
        u_h = cv.getTrackbarPos('Upper_Hue','Color-Tracker')
        u_s = cv.getTrackbarPos('Upper_Sat','Color-Tracker')
        u_v = cv.getTrackbarPos('Upper_Val','Color-Tracker')

        # lower_color = np.array([110,50,50])
        # upper_color = np.array([130,255,255])
        lower_color = np.array([l_h, l_s,l_v])
        upper_color = np.array([u_h, u_s, u_v])

        mask = cv.inRange(hsv, lower_color, upper_color)
        res =  cv.bitwise_or(img, img, mask=mask)

        cv.imshow('Frame',img)
        cv.imshow('Mask',mask)
        cv.imshow('Res',res)
        if cv.waitKey(0) & 0xff == ord('q'):
            
            cv.destroyAllWindows()


def nothing(x):
    pass


cv.namedWindow('Color-Tracker')
cv.createTrackbar('Lower_Hue','Color-Tracker',0,180, nothing)
cv.createTrackbar('Lower_Sat','Color-Tracker',0,255, nothing)
cv.createTrackbar('Lower_Val','Color-Tracker',0,255, nothing)
cv.createTrackbar('Upper_Hue','Color-Tracker',255,255, nothing)
cv.createTrackbar('Upper_Sat','Color-Tracker',255,255, nothing)
cv.createTrackbar('Upper_Val','Color-Tracker',180,180, nothing)

findObject('Data/ball_1.jpg')


#Tennis Ball = 29-11-91