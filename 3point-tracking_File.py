import cv2
import numpy as np
import pyzed.sl as sl
from rigid_transform_3D import rigid_transform_3D
import matplotlib.pyplot as plt

import Camera
from ValueHSV import ValueHSV

#Si crea la finestra per gestire la risoluzione, e 3 pulsanti, uno per ogni punto
titleWindow='JawTracking'
cv2.namedWindow(titleWindow, cv2.WINDOW_NORMAL)
cv2.resizeWindow(titleWindow, 500,100)

#Vettore con 3 colori da assegnare ad ogni punto
global rgb, res
rgb=[(0,0,255),(255,0,0),(0,255,0)]

#Variabile che contiene i valori HSV di ogni punto
value_hsv=np.zeros((3,6))

#Funzioni Callback per ogni bottone, una per ogni punto, che riempiono la matrice value_hsv
def DefinePoint1(x):
    global value_hsv, track_window1 #finestra da cui il camshift principale può iniziare
    #In questa funzione si può fare un rettangolo con il mouse e si prendono i valori hsv all interno, inoltre
    #viene aperta una finestra per regoalre i min e max dei valori hsv cosi da isolare meglio il punto, poichè
    #viene mostrata un anteprima dell'isolmento del punto.
    min_hue, min_sat, min_val, max_hue, max_sat, max_val, track_window1 = ValueHSV('Point1', rgb[0], filepath, res)
    value_hsv[0] = [min_hue, min_sat, min_val, max_hue, max_sat, max_val]


def DefinePoint2(x):
    global value_hsv, track_window2 
    min_hue, min_sat, min_val, max_hue, max_sat, max_val, track_window2  = ValueHSV('Point2', rgb[1], filepath, res)
    value_hsv[1] = [min_hue, min_sat, min_val, max_hue, max_sat, max_val]


def DefinePoint3(x):
    global value_hsv, track_window3
    min_hue, min_sat, min_val, max_hue, max_sat, max_val, track_window3 = ValueHSV('Point3', rgb[2], filepath, res)
    value_hsv[2] = [min_hue, min_sat, min_val, max_hue, max_sat, max_val]

def nothing(x):
    pass

#Si definisce una varibile legata alla risoluzione che avranno le immagini su cui si lavora
def resolution(x):
    global res
    if x == 0:
        res = 'HD2K'
    if x == 1:
        res = 'HD1080'
    if x == 2:
        res = 'HD720'
    if x == 3:
        res = 'VGA'
    pass

res= None
#Si creno i trackbar per ogni punto cosi di può avviare il ciclo per ottenere i valori hsv
cv2.createTrackbar('resolution  HD2K=0  VGA=3        ', titleWindow, 0, 3, resolution)
cv2.createTrackbar('Point1', titleWindow, 0, 1, DefinePoint1)
cv2.createTrackbar('Point2', titleWindow, 0, 1, DefinePoint2)
cv2.createTrackbar('Point3', titleWindow, 0, 1, DefinePoint3)

if __name__ == '__main__':
    global track_window1, track_window2, track_window3
    ret_t=0
    t=0
    tracking = False
    B=np.zeros((3,3))

    #Si definisce il percorso del video in formato .svo ottenuto dal ZED Explorer.
    #il file .svo permette di richiamare gli stessi metodi come se fosse collegata la camera
    filepath = 'D:\Documents\ZED\HD2K_Test.svo' 
    print("Reading SVO file: {0}".format(filepath))

    ok=0
    #Il ciclo resta qui fintanto che non si preme il tasto ok, cosi si possono ottenere i valori HSV
    while ok==0:
        k=cv2.waitKey(1) 
        if k == 13:
            break

    #Creo l'oggetto Camera a cui passo il percorso del video
    zed = Camera.Camera(filepath)
    #Metodo che richiama il necessario per avviare la camera
    zed.OpenCamera()

    runtime=zed.runtime  
    #numero frame del video
    frame_max = 0
    one=0 #varibile per inizializzare i dati del tracking
    b=0 #variabile per scrivere la prima riga dei dati 
    for i in range(frame_max):
        err = zed.zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:  #se c'è un errore nella camera si ferma
            #Si prende il frame che è già statto corretto dalla distorsione delle camera
            l_img, r_img = zed.Image(res)
            img=l_img

            #Finestra che mostra l'immagine
            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            
            #Si modifica la gamma colori da BGRA a BGR, quindi si toglie il paramentro A legato all intensita
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            #Si costruiscono 3 immagini da questa, dove per ognuna vengono posti neri i pixel con valori HSV al di fuori di quelli in value_hsv
            lower_bound= value_hsv[:,0:3]
            upper_bound = value_hsv[:,3:6]
            #da BGR a HSV
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            #Immagini con i pixel neri
            mask_img1 = cv2.inRange(hsv_img, lower_bound[0] , upper_bound[0])
            mask_img2 = cv2.inRange(hsv_img, lower_bound[1] , upper_bound[1])
            mask_img3 = cv2.inRange(hsv_img, lower_bound[2] , upper_bound[2])
                
            #Unione dei pixel colorati e neri
            hsv_mask_img1 = np.bitwise_and(img, mask_img1[:, :, np.newaxis]) 
            hsv_mask_img2 = np.bitwise_and(img, mask_img2[:, :, np.newaxis]) 
            hsv_mask_img3 = np.bitwise_and(img, mask_img3[:, :, np.newaxis]) 

            #Applicazione algoritmo CamShift che racchiude l'unico elemento non nero in un ellisse
            #restituendo il centro dell'ellisse in pixel e la sua dimenisone
            
            #imposta i criteri di terminazione per usare CamShift
            l_term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) 
            
            l_track_box1, l_track_window1 = cv2.CamShift(mask_img1, l_track_window1, l_term_crit)
            l_track_box2, l_track_window2 = cv2.CamShift(mask_img2, l_track_window2, l_term_crit)
            l_track_box3, l_track_window3 = cv2.CamShift(mask_img3, l_track_window3, l_term_crit)

            #Vengono rappresentati gli ellissi sull'immagine img
            cv2.ellipse(img, l_track_box1, rgb[0], 2)
            cv2.ellipse(img, l_track_box2, rgb[1], 2)
            cv2.ellipse(img, l_track_box3, rgb[2], 2)

            cv2.imshow('Tracking', img) 

            #richiama il codice per ottenere la nuvola di punti dell'immagine
            point_cloud = zed.PointCloud(res)

            #Ottengo X, Y, Z di ogni punto da tracciare, richiamando get_value per ogni 
            #centro degli ellissi ottenuti dall'immagine sinistra 
            err1, point1 = zed.point_cloud.get_value(int(l_track_box1[0][0]), int(l_track_box1[0][1]))
            err2, point2 = zed.point_cloud.get_value(int(l_track_box2[0][0]), int(l_track_box2[0][1]))
            err3, point3 = zed.point_cloud.get_value(int(l_track_box3[0][0]), int(l_track_box3[0][1]))

            if np.isnan(point1[2]) != True :
                X1=point1[0]
                Y1=point1[1]
                Z1=point1[2]
            
            if np.isnan(point2[2]) != True :
                X2=point2[0]
                Y2=point2[1]
                Z2=point2[2]
            
            if np.isnan(point3[2]) != True :
                X3=point3[0]
                Y3=point3[1]
                Z3=point3[2]

            #se viene premuto il tasto t avvia il tracking, se la prima volta inizializza le variabili
            if tracking and one==0:
                trans = centroid_A 
                cent_B = centroid_B 

                track_p1 = np.array([[X1], [Y1], [Z1]])
                track_p2 = np.array([[X2], [Y2], [Z2]])
                track_p3 = np.array([[X3], [Y3], [Z3]])
                T=trans
                t1=ret_t

                one+=1

            #se tutti e tre i punti ottenuto le tre coordinate, calcola il vettore traslazione e la matrice rotazione
            if np.isnan(point1[2]) != True and  np.isnan(point2[2]) != True and np.isnan(point3[2]) != True:
                B = np.array([[X1,X2,X3], [Y1,Y2,Y3], [Z1,Z2,Z3]])
                
                R, ret_t, centroid_A, centroid_B = rigid_transform_3D(A, B)

            A = B

            #se il tracking è avviato, registra le coordinate dei tre punti, il vettore traslazione e la matrice di rotazione di ogni frame
            if tracking:  
                trans += ret_t
                T = np.append(T, [trans[0],trans[1],trans[2]], axis=1)              
                print(i)

                cent_B = np.append(cent_B, [centroid_B[0], centroid_B[1], centroid_B[2]], axis=1)      

                track_p1 = np.append(track_p1, [[X1], [Y1], [Z1]], axis=1)
                track_p2 = np.append(track_p2, [[X2], [Y2], [Z2]], axis=1)
                track_p3 = np.append(track_p3, [[X3], [Y3], [Z3]], axis=1)

                data = str(trans[0]) + ' , ' + str(trans[1]) + ' , ' + str(trans[2]) + ' , ' + str(R[0][0]) \
                + ' , ' + str(R[0][1]) + ' , ' + str(R[0][2]) + ' , ' + str(R[1][0]) + ' , ' + str(R[1][1])\
                + ' , ' + str(R[1][2]) + ' , ' + str(R[2][0]) + ' , ' + str(R[2][1]) + ' , ' + str(R[2][2])   
                #salva i dati in un file
                if b==0:
                    file_data = open("D:\\Desktop\JawTracking\src\Data.csv", "w")
                    file_data.write('tX    tY     tZ     R1      R2     R2      R4    R5      R6   R7   R8   R9  ')
                    file_data.close()
                else:
                    file_data = open("D:\\Desktop\JawTracking\src\Data.csv", "a")
                    file_data.write(data)
                    file_data.write('\n')
                    file_data.close()
                b+=1

        #Raggiunti un certo numero di frame di registrazione, mostra le traiettorie registrate
        if i==200:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_ylim([0, 400])

            ax.plot(track_p1[0][:],track_p1[1][:],track_p1[2][:], color='red')
            ax.plot(track_p2[0][:],track_p2[1][:],track_p2[2][:], color='blue')
            ax.plot(track_p3[0][:],track_p3[1][:],track_p3[2][:], color='green')
            ax.plot(cent_B[0][:],cent_B[1][:],cent_B[2][:], color='black' )
            ax.plot(T[0][:],T[1][:],T[2][:])
            break
        
        
        c=cv2.waitKey(1) 
        if c == 27:
            break  
        if c == 116:
             tracking = True 
    
    plt.show()
    print('fine')