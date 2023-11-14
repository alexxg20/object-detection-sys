import cv2 as cv
import numpy as np
import os
from operator import itemgetter
import math
import datetime

# read from webcam
vid = cv.VideoCapture('C:/Users/2alex/OneDrive/Pictures/Álbum de cámara/Video_Campo1.mp4')
field = cv.imread('C:/Users/2alex/OneDrive/Escritorio/python/images/soccer-field.jpg')

if (vid.isOpened() == False): 
  print("Unable to read camera feed")

def empty(a):
    pass

def ratio(cnt, a):
   
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    width = math.sqrt(pow(box[0][0]-box[1][0], 2) + pow(box[0][1]-box[1][1], 2))
    height = math.sqrt(pow(box[0][0]-box[3][0], 2) + pow(box[0][1]-box[3][1], 2))
    # sort by value
    little, big =  sorted((width, height))
    # calculate ratio 
    ratio = big/little
    # check ratio
    if 5 >= ratio > 1.5 and a == 1:
        return True
    elif 1.8 >= ratio > 0.8 and a == 2:
        return True
    else:
        print('ERROR: Aspect ratio does not match')
        return False

def detect_aruco(image):

    #print(" Detecting markers...")
    a = datetime.datetime.now()
    
    arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    arucoParams =  cv.aruco.DetectorParameters()
    
    detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)
    (corners, ids, rejected) = detector.detectMarkers(image)
    corners_fr = []
    num = 0
    if ids != '[]':
        for id in range(6):
            if id in ids:
                index = list(ids).index(id)
                corners_fr.append(corners[index])
                num += 1
            else: continue
    else:
        points = []
        return points, ids
    
    if len(corners_fr) >= 4:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners4 = markerCorner.reshape((4, 2))         
            
    else:
        print('\tNumber of markers ' + str(len(corners_fr)) + '. Skipping image...')
        return
    
    # sort ids by number
    ids = np.sort(ids)  

    # Save sorted points
    points = []
    for i in range(6):
        try:
            if list(ids).index(i) or list(ids).index(i) == 0:
                markerCorner = corners_fr[list(ids).index(i)]
                corners4 = markerCorner.reshape((4, 2))
                points.append(corners4)
            else:
                continue
        except:
            print(f"Index {i} does not exist")
            continue
    
    b = datetime.datetime.now()
    c = b - a
    #print(f" ... {c.microseconds/1000} ms")
    print(str(points) + str(ids))
    return points, ids

def transform_image(points, image, ids, coordinate_trans_p, coordinate_trans_b):
    
    pts1 = np.float32([[-16, -16], # id0
                      [320, -16], # id1
                      [656, -16], #id2
                      [656, 432], # id3
                      [320, 432], # id4
                      [-16, 432]]) # id5
    '''
    pts1 = np.float32([[0, 0], # id0
                      [320, 0], # id1
                      [640, 0], #id2
                      [640, 416], # id3
                      [320, 416], # id4
                      [0, 416]]) # id5
    '''
    '''
    ptslist = []
    for corners4 in points:
        (topLeft, topRight, bottomRight, bottomLeft) = corners4
        ptslist.append(topLeft)
        ptslist.append(topRight)
        ptslist.append(bottomLeft)
        ptslist.append(bottomRight)
    '''                
        
    pts2 = []
    for aruco_num in range (len(points)):
        sum_x = 0
        sum_y = 0
        for aruco_point in range(4):
            for axis in range(2):
                if axis == 0:
                    sum_x += points[aruco_num][aruco_point][axis]
                else:
                    sum_y += points[aruco_num][aruco_point][axis]
        average_x = sum_x/4
        average_y = sum_y/4
        pts2.append(average_x)
        pts2.append(average_y)
    
    pt1 = []
    ids = [x for x in ids if x <= 6]
    for num in range(len(ids)):
        pt1.append(pts1[ids[num]])
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pts2, dtype=np.float32)
    pt2 = pt2.reshape((len(ids),2))
    
    # Apply Perspective Transform Algorithm
    matrix = cv.findHomography(pt2, pt1, 0)
    H = np.array(matrix[0])
    print(H)
    result = cv.warpPerspective(image, H, (640, 416))

    # transform player coordinates
    if coordinate_trans_p != []:
        coordinate_tmp = np.matmul(H, coordinate_trans_p)
        coordinate_trans_p = np.array([(coordinate_tmp[0]/coordinate_tmp[2]), (coordinate_tmp[1]/coordinate_tmp[2])])
    
    if coordinate_trans_b != []:
        print("ppppppppppppppp")
        coordinate_tmp_b = np.matmul(H, coordinate_trans_b)
        coordinate_trans_b = np.array([(coordinate_tmp_b[0]/coordinate_tmp_b[2]), (coordinate_tmp_b[1]/coordinate_tmp_b[2])])
        '''
        cv.drawMarker(field, (int(coordinate_trans[0]), int(coordinate_trans[1])), color=[0, 0, 0], thickness=3, 
        markerType= cv.MARKER_TILTED_CROSS, line_type=cv.LINE_AA,
        markerSize=10)
        '''

    return result, coordinate_trans_p, coordinate_trans_b

# create window for trackbar
cv.namedWindow("HSV Values (P)")
cv.resizeWindow("HSV Values (P)", 400, 200)

cv.namedWindow("HSV Values (B)")
cv.resizeWindow("HSV Values (B)", 400, 200)

# threshold values for the player
cv.createTrackbar("HUE Min (P)", "HSV Values (P)", 8, 40 , empty)
cv.createTrackbar("HUE Max (P)", "HSV Values (P)", 40,40 , empty)
cv.createTrackbar("SAT Min (P)", "HSV Values (P)", 25,255 , empty)
cv.createTrackbar("SAT Max (P)", "HSV Values (P)", 255,255 , empty)

# threshold values for the ball
cv.createTrackbar("HUE Min (B)", "HSV Values (B)", 150, 175 , empty)
cv.createTrackbar("HUE Max (B)", "HSV Values (B)", 175,175 , empty)
cv.createTrackbar("SAT Min (B)", "HSV Values (B)", 110,255 , empty)
cv.createTrackbar("SAT Max (B)", "HSV Values (B)", 255,255 , empty)

# initialize
coordinate_old_p = []
coordinate_old_b = []
coordinate_trans_p = []
coordinate_trans_b = []
count = 0

while(vid.isOpened()):
    # capture each video frame
    ret, frame = vid.read()
    frame = cv.resize(frame, (960, 540)) 
    # apply gaussian blur
    blurred = cv.GaussianBlur(frame, (3, 3), 0)
    # convert to hsv
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    h_min_p = cv.getTrackbarPos("HUE Min (P)", "HSV Values (P)")
    h_max_p = cv.getTrackbarPos("HUE Max (P)", "HSV Values (P)")
    s_min_p = cv.getTrackbarPos("SAT Min (P)", "HSV Values (P)")
    s_max_p = cv.getTrackbarPos("SAT Max (P)", "HSV Values (P)")

    h_min_b = cv.getTrackbarPos("HUE Min (B)", "HSV Values (B)")
    h_max_b = cv.getTrackbarPos("HUE Max (B)", "HSV Values (B)")
    s_min_b = cv.getTrackbarPos("SAT Min (B)", "HSV Values (B)")
    s_max_b = cv.getTrackbarPos("SAT Max (B)", "HSV Values (B)")


    # define color range
    yellow_lower = np.array([h_min_p, s_min_p, 0], np.uint8)
    yellow_upper = np.array([h_max_p, s_max_p, 255], np.uint8)

    orange_lower = np.array([h_min_b, s_min_b, 0], np.uint8)
    orange_upper = np.array([h_max_b, s_max_b, 255], np.uint8)

    # define mask
    mask_p = cv.inRange(hsv, yellow_lower, yellow_upper)

    mask_b = cv.inRange(hsv, orange_lower, orange_upper)
    
    path = 'C:/Users/2alex/OneDrive/Escritorio/python/images'
    cv.imwrite(os.path.join(path , 'opencv_image1.jpg'), mask_p)
    cv.imwrite(os.path.join(path , 'opencv_image2.jpg'), frame)

    # Morphological Transform: Erosion, Dilation (Player)
    kernel = np.ones((3, 3), "uint8")
    mask_p = cv.erode(mask_p, kernel)
    cv.imwrite(os.path.join(path , 'opencv_image_pl3.jpg'), mask_p)
    kernel = np.ones((5, 5), "uint8")
    mask_p = cv.dilate(mask_p, kernel)
    cv.imwrite(os.path.join(path , 'opencv_image_pl4.jpg'), mask_p)

    res_yellow = cv.bitwise_and(frame, frame, 
                              mask = mask_p)
    # Creating contour to track yellow color
    contours_p, hierarchy = cv.findContours(mask_p,
                                           cv.RETR_TREE,
                                           cv.CHAIN_APPROX_SIMPLE)
    
    # Morphological Transform: Erosion, Dilation (Ball)
    kernel = np.ones((3, 3), "uint8")
    mask_b = cv.erode(mask_b, kernel)
    cv.imwrite(os.path.join(path , 'opencv_image_ball3.jpg'), mask_b)
    kernel = np.ones((5, 5), "uint8")
    mask_b = cv.dilate(mask_b, kernel)
    cv.imwrite(os.path.join(path , 'opencv_image_ball4.jpg'), mask_b)

    res_orange = cv.bitwise_and(frame, frame, 
                              mask = mask_b)
    # Creating contour to track orange color
    contours_b, hierarchy = cv.findContours(mask_b,
                                           cv.RETR_TREE,
                                           cv.CHAIN_APPROX_SIMPLE)
    
    p_detected = False  # player is detected
    if contours_p  != "()":
        for contour in contours_p: 
            area = cv.contourArea(contour)
            if(area > 400) and p_detected == False:
                #print(contour)
                if ratio(contour, 1):
                    x, y, w, h = cv.boundingRect(contour)
                
                    frame = cv.rectangle(frame, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2)
                
                    cv.putText(frame, "Yellow Color", (x, y),
                            cv.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))
                    x1_p = (x + (w/2)) 
                    y1_p = (y + h)
                    p_detected = True

    b_detected = False  # ball is detected
    if contours_b  != "()":
        for contour in contours_b: 
            area = cv.contourArea(contour)
            if(area > 200) and (area < 1000) and b_detected == False:
                #print(contour)
                if ratio(contour, 2):
                    x, y, w, h = cv.boundingRect(contour)

                    frame = cv.rectangle(frame, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2)
                
                    cv.putText(frame, "Orange Color", (x, y),
                            cv.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))
                    x1_b = (x + (w/2)) 
                    y1_b = (y + h)
                    b_detected = True
    else:
        print("Ball not detected")
    
    # check if player has been detected
    if p_detected == True:
        print(str(x1_p)+str("fffffffff"))
        print(str(y1_p)+str("fffffffff"))
        cartesian_coordinate_p = [x1_p, y1_p, 1]
    else:
        cartesian_coordinate_p = []
        print("False" + str(p_detected))
    
    # check if ball has been detected
    if b_detected == True:
        cartesian_coordinate_b = [x1_b, y1_b, 1]
    else:
        cartesian_coordinate_b = []
        print("False" + str(p_detected))
    
    # detect aruco positions
    try:
        points, ids = detect_aruco(frame.copy())
    except:
        print("points not detected")
        points = []

    # transform frame and object coordinates
    if (points != []):
       transformed_frame, coordinate_trans_p, coordinate_trans_b = transform_image(points, frame, ids, cartesian_coordinate_p, cartesian_coordinate_b)    

    if count >= 4:
        if coordinate_trans_p != []:
            #print(coordinate_trans)
            print("lllllllllllllllllllll")
            print(coordinate_old_p)
            print(coordinate_trans_p)
            if coordinate_old_p == []:
                cv.line(field, (int(coordinate_trans_p[0]), int(coordinate_trans_p[1])), (int(coordinate_trans_p[0]), int(coordinate_trans_p[1])), (0,0,0), 3)
                coordinate_old_p = coordinate_trans_p
            elif coordinate_old_p != [] and coordinate_trans_p != [] and coordinate_trans_p[0] < ((coordinate_old_p[0] + 50) and 
                                                                     coordinate_trans_p[1] < (coordinate_old_p[1] + 50) and 
                                                                    coordinate_trans_p[0] > (coordinate_old_p[0] - 50) and
                                                                      coordinate_trans_p[1]) > (coordinate_old_p[1] - 50):
                
                print("fffffffffffffffffffffffff")
                #print(coordinate_trans_p)
                #print(coordinate_old_p)
                cv.line(field, (int(coordinate_old_p[0]), int(coordinate_old_p[1])), (int(coordinate_trans_p[0]), int(coordinate_trans_p[1])), (0,0,0), 3)
                coordinate_old_p = coordinate_trans_p
            count = 0

        if coordinate_trans_b != []:
            #print(coordinate_trans)
            print("pppppppppppppppppppppp")
            print(coordinate_old_b)
            print(coordinate_trans_b)
            if coordinate_old_b == []:
                cv.line(field, (int(coordinate_trans_b[0]), int(coordinate_trans_b[1])), (int(coordinate_trans_b[0]), int(coordinate_trans_b[1])), (0,0,0), 3)
                coordinate_old_b = coordinate_trans_b
            elif coordinate_trans_b[0] < ((coordinate_old_b[0] + 50) and 
                                                                     coordinate_trans_b[1] < (coordinate_old_b[1] + 50) and 
                                                                    coordinate_trans_b[0] > (coordinate_old_b[0] - 50) and
                                                                      coordinate_trans_b[1]) > (coordinate_old_b[1] - 50):
                print("kkkkkkkkkkkkkkkkkkkkkkk")
                cv.line(field, (int(coordinate_old_b[0]), int(coordinate_old_b[1])), (int(coordinate_trans_b[0]), int(coordinate_trans_b[1])), (0,0,0), 3)
                coordinate_old_b = coordinate_trans_b
            count = 0
    cv.imshow("position transformed", field)
    count += 1

    # original image with bounding box
    cv.imshow("Color Detection in Real Time", frame)


    # segmented image
    cv.imshow("segmented image_p", mask_p)
    cv.imshow("segmented image_b", mask_b)
    

    # write a given frame to a file
    k = cv.waitKey(1)
    if k%256 == 32:
        # SPACE pressed
        path = 'C:/images/'
        cv.imwrite(os.path.join(path , 'opencv_image.jpg'), frame)
        
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()