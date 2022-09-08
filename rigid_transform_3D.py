#!/usr/bin/python

import numpy as np

# Input: si aspetta una matrice 3xN di punti
# Returns R,t
# R = 3x3 Matrice di rotazione
# t = 3x1 vettore di traslazione

def rigid_transform_3D(A, B):
    assert A.shape == B.shape  #se le dim di A sono uguali alle dim di B puoi continuare ad eseguire

    #Controllo dimensioni matrici
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # trova le medie delle colonne
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # crea array 3x1
    centroid_A = centroid_A.reshape(-1, 1)  
    centroid_B = centroid_B.reshape(-1, 1)

    # Sottrae ai punti i centroidi
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)  #costruisco la matrice H che conterrò la rotazione tra A e B

    # trova la rotazione
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T 
    
    # caso speciale di riflessione
    if np.linalg.det(R) < 0: 
        
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t, centroid_A, centroid_B