
import pyzed.sl as sl

import numpy as np
import cv2

class Camera():

    def __init__(self, a = None):
        # Create a ZED camera object

        if a == None:
            self.init = sl.InitParameters()
            self.init.camera_resolution = sl.RESOLUTION.HD2K
            self.init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
            self.init.depth_mode = sl.DEPTH_MODE.ULTRA

        else:
            input_type = sl.InputType()
            input_type.set_from_svo_file(a)
            self.init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
            self.init.camera_resolution = sl.RESOLUTION.HD2K
            self.init.depth_mode = sl.DEPTH_MODE.ULTRA
            self.init.coordinate_units = sl.UNIT.MILLIMETER
            self.init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
            self.init.depth_maximum_distance = 1000
            self.init.depth_minimum_distance = 300

        self.zed = sl.Camera()

    def OpenCamera(self):
        
        # Open the camera
        err = self.zed.open(self.init)
        if err != sl.ERROR_CODE.SUCCESS :
            print(repr(err))
            self.zed.close()
            exit(1)

        self.runtime = sl.RuntimeParameters()        
        
        # Prepare new image size to retrieve half-resolution images
        self.image_size = self.zed.get_camera_information().camera_resolution
        
        # Declare your sl.Mat matrices
        self.image_zed_l = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.image_zed_r = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.depth_image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.point_cloud = sl.Mat()

    def Image(self, b = None):
        if b == None:
            # Retrieve the left image Undistort
            self.zed.retrieve_image(self.image_zed_l, sl.VIEW.LEFT, sl.MEM.CPU, self.image_size)
            l_img = self.image_zed_l.get_data() #questo trasforma l'immagine che ha preso in una matrice come fa videocapture
            
            # Retrieve the right image Undistort
            self.zed.retrieve_image(self.image_zed_r, sl.VIEW.RIGHT, sl.MEM.CPU, self.image_size)
            r_img = self.image_zed_r.get_data()
        
        if b == 'HD2K':
            resolution = sl.Resolution(2208,1242)

            # Retrieve the left image Undistort
            self.zed.retrieve_image(self.image_zed_l, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
            l_img = self.image_zed_l.get_data() #questo trasforma l'immagine che ha preso in una matrice come fa videocapture
            
            # Retrieve the right image Undistort
            self.zed.retrieve_image(self.image_zed_r, sl.VIEW.RIGHT, sl.MEM.CPU, resolution)
            r_img = self.image_zed_r.get_data()
        
        if b == 'HD1080':
            resolution = sl.Resolution(1920,1080)

            # Retrieve the left image Undistort
            self.zed.retrieve_image(self.image_zed_l, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
            l_img = self.image_zed_l.get_data() #questo trasforma l'immagine che ha preso in una matrice come fa videocapture
            
            # Retrieve the right image Undistort
            self.zed.retrieve_image(self.image_zed_r, sl.VIEW.RIGHT, sl.MEM.CPU, resolution)
            r_img = self.image_zed_r.get_data()

        if b == 'HD720':
            resolution = sl.Resolution(1280,720)

            # Retrieve the left image Undistort
            self.zed.retrieve_image(self.image_zed_l, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
            l_img = self.image_zed_l.get_data() #questo trasforma l'immagine che ha preso in una matrice come fa videocapture
            
            # Retrieve the right image Undistort
            self.zed.retrieve_image(self.image_zed_r, sl.VIEW.RIGHT, sl.MEM.CPU, resolution)
            r_img = self.image_zed_r.get_data()

        if b == 'VGA':
            resolution = sl.Resolution(672,376)

            # Retrieve the left image Undistort
            self.zed.retrieve_image(self.image_zed_l, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
            l_img = self.image_zed_l.get_data() #questo trasforma l'immagine che ha preso in una matrice come fa videocapture
            
            # Retrieve the right image Undistort
            self.zed.retrieve_image(self.image_zed_r, sl.VIEW.RIGHT, sl.MEM.CPU, resolution)
            r_img = self.image_zed_r.get_data()

        return l_img, r_img 
    
    def Depth(self, c = None):
        if c == None:
            self.zed.retrieve_image(self.depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, self.image_size)
            depth_image = self.depth_image_zed.get_data()
        
        if c == 'VGA':
            resolution = sl.Resolution(672,376)

            self.zed.retrieve_image(self.depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, resolution)
            depth_image = self.depth_image_zed.get_data()
        
        return depth_image

    def PointCloud(self, d = None ):
        if d == None:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.image_size)
        
        if d == 'VGA':
            resolution = sl.Resolution(672,376)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution)


    def Q(self, height, width):
        res = sl.Resolution(height, width)
        R = self.zed.get_camera_information(res).calibration_parameters.R
        T = self.zed.get_camera_information(res).calibration_parameters.T

        R1 = np.zeros(shape=(3,3))
        R2 = np.zeros(shape=(3,3))
        P1 = np.zeros(shape=(3,3))
        P2 = np.zeros(shape=(3,3))
        Q = np.zeros(shape=(4,4))

        self.DistParam(height, width)

        cv2.stereoRectify(l_cam_mtx,  l_dist_coeffs, r_cam_mtx, r_dist_coeffs, (height, width), R, T, R1, R2, P1, P2, Q, alpha=-1, newImageSize=(0,0))

        return Q

    def DistParam(self,height, width):
        global  l_cam_mtx, r_cam_mtx,  l_dist_coeffs, r_dist_coeffs, fx_l, fy_l, cx_l, cy_l, fx_r, fy_r, cx_r, cy_r

        res = sl.Resolution(height, width)
        fx_l = self.zed.get_camera_information(res).calibration_parameters.left_cam.fx
        fy_l = self.zed.get_camera_information(res).calibration_parameters.left_cam.fy
        cx_l = self.zed.get_camera_information(res).calibration_parameters.left_cam.cx
        cy_l = self.zed.get_camera_information(res).calibration_parameters.left_cam.cy

        l_cam_mtx = np.array([[fx_l,   0., cx_l],
                            [  0., fy_l, cy_l],
                            [  0.,   0.,  1. ]])
            
        l_dist_coeffs = self.zed.get_camera_information(res).calibration_parameters.left_cam.disto

        fx_r = self.zed.get_camera_information(res).calibration_parameters.right_cam.fx
        fy_r = self.zed.get_camera_information(res).calibration_parameters.right_cam.fy
        cx_r = self.zed.get_camera_information(res).calibration_parameters.right_cam.cx
        cy_r = self.zed.get_camera_information(res).calibration_parameters.right_cam.cy

        r_cam_mtx = np.array([[fx_r,   0., cx_r],
                            [  0., fy_r, cy_r],
                            [  0.,   0.,  1. ]])
                            
        r_dist_coeffs = self.zed.get_camera_information(res).calibration_parameters.right_cam.disto




