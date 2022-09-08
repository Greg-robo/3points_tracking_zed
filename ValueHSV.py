import cv2
import numpy as np
import pyzed.sl as sl

import Camera

def nothing(x):
    pass

#Funzione per visualizzare il rettangolo della regione di interesse ROI
def roi_click_event(event, x, y, flags, params):
    global draw_hsv_roi
    global show_hsv_roi_cnt  
    global x1, y1, x2, y2
    global Min_hue, Min_sat, Min_val, Max_hue, Max_sat, Max_val
    global img

    # Controllo evento tasto sinistro mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        draw_hsv_roi = True
    
    # Controllo se prima era gia stato premuto e viene mantenuto
    if draw_hsv_roi:
        # Contatore per quanto tempo il rettangolo doverebbe stare sullo schermo dopo il rilascio del tasto
        show_hsv_roi_cnt = 15 
        
        x2, y2 = x, y
        
    # Controllo se il tasto è stato rilasciato
    if event == cv2.EVENT_LBUTTONUP:
        # Controllo dei contorno
        if y1 > y2:
            tmp = y2
            y2 = y1
            y1 = tmp
            
        if x1 > x2:
            tmp = x2
            x2 = x1
            x1 = tmp
            
        if y1 == y2:
            y2 += 1
            
        if x1 == x2:
            x2 += 1
            
        roi_bgr = img[y1:y2, x1:x2]
        
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        #si prendono i valori HSV
        Min_hsv = np.amin((np.amin(roi_hsv, 1)), 0)
        Max_hsv = np.amax((np.amax(roi_hsv, 1)), 0)
        
        tolerance = 3
        
        draw_hsv_roi = False

        cv2.setTrackbarPos('min_hue', 'roi_selector_trackbars', Min_hsv[0] - tolerance)
        cv2.setTrackbarPos('min_sat', 'roi_selector_trackbars', Min_hsv[1] - tolerance) 
        cv2.setTrackbarPos('min_val', 'roi_selector_trackbars', Min_hsv[2] - tolerance) 
        cv2.setTrackbarPos('max_hue', 'roi_selector_trackbars', Max_hsv[0] + tolerance) 
        cv2.setTrackbarPos('max_sat', 'roi_selector_trackbars', Max_hsv[1] + tolerance) 
        cv2.setTrackbarPos('max_val', 'roi_selector_trackbars', Max_hsv[2] + tolerance)

def ValueHSV( name, rgb, filepath, res):
    #Viene creata la finestra per i trackbar dei valori
    cv2.namedWindow('roi_selector_trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('roi_selector_trackbars', 700, 100)

    cv2.createTrackbar('min_hue', 'roi_selector_trackbars', 0,   180, nothing)
    cv2.createTrackbar('max_hue', 'roi_selector_trackbars', 180, 180, nothing)
    cv2.createTrackbar('min_sat', 'roi_selector_trackbars', 0,   255, nothing)
    cv2.createTrackbar('max_sat', 'roi_selector_trackbars', 255, 255, nothing)
    cv2.createTrackbar('min_val', 'roi_selector_trackbars', 0,   255, nothing)
    cv2.createTrackbar('max_val', 'roi_selector_trackbars', 255, 255, nothing)

    global img
    global show_hsv_roi_cnt
    global draw_hsv_roi
    global x1, y1, x2, y2

    draw_hsv_roi = False
    show_hsv_roi_cnt = 0
    x1, y1, x2, y2 = 0, 0, 0, 0

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)   
    cv2.setMouseCallback(name, roi_click_event)
    L_track_window = (0, 0, 100, 100) # x, y, width, height

    zed = Camera.Camera(filepath)
    zed.OpenCamera()

    runtime=zed.runtime
    while True:
        err = zed.zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:

            l_img, r_img = zed.Image(res)
            img = l_img

             # Recombine Img
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            ### STEP 2: HSV FILTERING ###
            Min_hue = cv2.getTrackbarPos('min_hue', 'roi_selector_trackbars')
            Max_hue = cv2.getTrackbarPos('max_hue', 'roi_selector_trackbars')
            Min_sat = cv2.getTrackbarPos('min_sat', 'roi_selector_trackbars')
            Max_sat = cv2.getTrackbarPos('max_sat', 'roi_selector_trackbars')
            Min_val = cv2.getTrackbarPos('min_val', 'roi_selector_trackbars')
            Max_val = cv2.getTrackbarPos('max_val', 'roi_selector_trackbars')
            
            lower_bound = (Min_hue, Min_sat, Min_val)
            upper_bound = (Max_hue, Max_sat, Max_val)

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_img = cv2.inRange(hsv_img, lower_bound , upper_bound)

            hsv_mask_img = np.bitwise_and(img, mask_img[:, :, np.newaxis]) 
            cv2.imshow(name, hsv_mask_img) 

            L_term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)  #imposta i criteri di terminazione per usare CamShift
            L_track_box, L_track_window = cv2.CamShift(mask_img, L_track_window, L_term_crit)

            cv2.ellipse(hsv_mask_img, L_track_box, rgb, 2)

            if show_hsv_roi_cnt >= 0:
                cv2.rectangle(hsv_mask_img, (x1, y1), (x2, y2), rgb, int(show_hsv_roi_cnt/3))
                if not draw_hsv_roi:
                    show_hsv_roi_cnt -= 5

        cv2.imshow(name, hsv_mask_img) 
        key = cv2.waitKey(1)
        if key == 13:
            cv2.destroyWindow(name)
            cv2.destroyWindow('roi_selector_trackbars')
            zed.zed.close()
            
            break
        if cv2.waitKey(1) == 27:
            break

    return Min_hue, Min_sat, Min_val, Max_hue, Max_sat, Max_val, L_track_window
