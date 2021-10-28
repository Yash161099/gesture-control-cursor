import cv2
import pyautogui
import numpy as np
import math
global prev
prev= [0, 0]
cap = cv2.VideoCapture(0)
def setCursorPos( current_p, prev_p):
	
	mouse_p = np.zeros(2)
	#print(current_p)
	#print(prev_p)
	if abs(current_p[0]-prev_p[0])>50 and abs(current_p[1]-prev_p[1])>50:
		mouse_p[0] = current_p[0] + .7*(prev_p[0]-current_p[0]) 
		mouse_p[1] = current_p[1] + .7*(prev[1]-current_p[1])
	else:
		mouse_p[0] = current_p[0] #+ .1*(prev_p[0]-current_p[0])
		mouse_p[1] = current_p[1] #+ .1*(prev_p[1]-current_p[1])
	
	return mouse_p

def centroid(max_contour_list):
    M = cv2.moments(max_contour_list)

    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return cx ,cy

    else:
        return None
def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float64)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float64)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None
while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:400, 100:400]
        
        
        cv2.rectangle(frame,(100,100),(400,400),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
         
    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.filter2D(mask, -1, kernel, mask)
          #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),0) 
        

        #mask = cv2.erode(mask, kernel, iterations= 5)
        
       # mask= cv2.merge((mask, mask, mask))
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
  
        #mask = cv2.bilateralFilter(mask, 21, 51, 51)
        
    #find contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   #find contour of max area(hand) and find the centroid
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        centroid_val = centroid(cnt)
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
    #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
    
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
    # l = no. of defects
        l=0
        
    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(roi,start, end, [0,255,0], 2)
            
            
        l+=1
        far_points = farthest_point(defects, cnt, centroid_val)
        cv2.circle(mask, far_points, 10, [255, 0, 0], -1)
        #print corresponding gestures which are in their ranges
        pyautogui.FAILSAFE = False
        mp = setCursorPos(far_points, prev)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<8:
                    cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                   
                else:
                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    pyautogui.moveTo((mp[0])*6,(mp[1])*4,0.3)
                    
        elif l==2 and arearatio>36:
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            pyautogui.click()
            
        elif l==3:
            cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==4:
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            pyautogui.scroll(-10)
        elif l==5:
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            im1 = pyautogui.screenshot('scr.png')
            #im1.save("c:\Users\Administrator\Desktop")
        elif l==6:
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        prev = far_points    
        #show the windows
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        pass
    
    k = cv2.waitKey(2) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    
