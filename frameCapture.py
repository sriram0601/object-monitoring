import cv2 as cv
import numpy as np

#Parameter 0 is to access the default camera

#Parameter -1 for some other platform to access the camera, To access the other camera that connect in device use intergers to 
#capture the video Eg: "cv.VideoCapture(2)" to access the 1st camera that connected in the device

class VideoFrame:
    def __init__(self, capture):
        self.capture = capture
    
    def webCam(self, vidCap = False):
        while (self.capture.isOpened()):
            ret, frame = self.capture.read() 
            

            
            # # print(f'{self.capture.get(cv.CAP_PROP_FRAME_WIDTH)} x {self.capture.get(cv.CAP_PROP_FRAME_HEIGHT)}')
            # grey = cv.cvtColor(frame,cv.COLOR_BGR2RGBA)
            # blurred = cv.GaussianBlur(grey, (7, 7), 0)
            # thresh = cv.adaptiveThreshold(blurred, 0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
            cv.imshow('Display Window',frame)

            if vidCap == True:
                save.write(frame)

            if cv.waitKey(1) & 0xFF==ord('q'):
                self.capture.release()
                save.release()
                cv.destroyAllWindows()
                break
            
    def detection(self):
        cv.namedWindow('Color-Tracker')
        cv.createTrackbar('Lower_Hue','Color-Tracker',0,180, nothing)
        cv.createTrackbar('Lower_Sat','Color-Tracker',0,255, nothing)
        cv.createTrackbar('Lower_Val','Color-Tracker',0,255, nothing)
        cv.createTrackbar('Upper_Hue','Color-Tracker',180,180, nothing)
        cv.createTrackbar('Upper_Sat','Color-Tracker',255,255, nothing)
        cv.createTrackbar('Upper_Val','Color-Tracker',255,255, nothing)
        while True:
            _,frame = self.capture.read()

            hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV_FULL)  #41-0-61

            l_h = cv.getTrackbarPos('Lower_Hue','Color-Tracker')
            l_s = cv.getTrackbarPos('Lower_Sat','Color-Tracker')
            l_v = cv.getTrackbarPos('Lower_Val','Color-Tracker')
            u_h = cv.getTrackbarPos('Upper_Hue','Color-Tracker')
            u_s = cv.getTrackbarPos('Upper_Sat','Color-Tracker')
            u_v = cv.getTrackbarPos('Upper_Val','Color-Tracker')

            lower_color = np.array([l_h, l_s,l_v])
            upper_color = np.array([u_h, u_s, u_v]) 

            mask = cv.inRange(hsv, lower_color, upper_color)
            res = cv.bitwise_and(frame,frame, mask=mask)

            cv.imshow('Frame',frame)
            cv.imshow('HSV',hsv)
            cv.imshow('RES',res)            
            
            save.write(res)
            if cv.waitKey(1) & 0xFF==ord('q'):
                self.capture.release()
                cv.destroyAllWindows()
            


def nothing(x):
    pass

cap = cv.VideoCapture(0)
forcc = cv.VideoWriter_fourcc(*'XVID')
save = cv.VideoWriter('output.avi', forcc, 20.0, (640, 480))
video = VideoFrame(cap)

video.detection()
