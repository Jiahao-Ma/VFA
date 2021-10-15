#TODO: define basic information of MultiviewC
#TODO: define annotation path, image path and calibration matrix
import numpy as np
from PIL import Image
import json, os, sys, cv2
from torchvision.datasets import VisionDataset
from torchvision import transforms
sys.path.append(os.getcwd())

intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml', 'intr_Camera7.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml', 'extr_Camera7.xml']

MULTIVIEWC_BBOX_LABEL_NAMES = ['Cow']

class MultiviewC(VisionDataset):
    def __init__(self, root = r'\Data\MultiviewC_dataset', # PATH of MultiviewC dataset
                       ann_root=r'annotations', 
                       img_root =r'images', 
                       calib_root=r'calibrations', 
                ) -> None:
        super().__init__(root)
        """
            Args:
                ann_root: annotation path
                img_root: image path
                calib_root: calibration path
            
            MultiviewC Map Setting:
                pivot_offset: x=1600  y=1600
                grid_size: w=3900 h=3900
                left_top coordxy:  (5500, 5500) 1600+3900 = 5500

            MultiviewC Camera Setting:
                theta_ref_global = theta_w_global + 90
                theta_c_global = theta_ref_global - R_z
                theta_c_global = theta_local + theta_ray
                                  
                theta_c_global = theta_w_global + 90 - R_z = theta_local + theta_ray
                
                R_z: the rotation angle of 7 cameras on Z-axis of the world coordinate in the farm 
                      [133.861435, -135.736145, -45.890991, 48.889431, 90.000084, 121.566719, 59.132477] 
                theta_ray: the angle between the ray from cammera center to objects' center 
                           and the y axis of camera.  (angle of camera coordinate) (-pi/2, pi/2)
                NOTICE: we need to keep theta_c_global in range [-pi, pi]
        """
        self.__name__ = 'MultiviewC'
        self.root = root
        # MultiviewC's unit is constant: centimeter (cm)  for calibration, location and dimension
        self.img_shape, self.world_shape = [720, 1280], [3900, 3900] # H, W, N_row, N_col
        self.num_cam, self.num_frame = 7, 560
        self.ann_root = os.path.join(root, ann_root)
        self.img_root = os.path.join(root, img_root)
        self.calib_root = os.path.join(root, calib_root)
        self.label_names = MULTIVIEWC_BBOX_LABEL_NAMES
        self.intrinsic_matrices, self.extrinsic_matrices, self.R_z = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam, root=self.calib_root) for cam in range(self.num_cam)])
     

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(1, 1 + self.num_cam)}
        for cam in range(1, 1+self.num_cam):
            img_folder = os.path.join(self.img_root, 'C{}'.format(cam))
            for fname in sorted(os.listdir(img_folder)):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(img_folder, fname)
        return img_fpaths

    def get_intrinsic_extrinsic_matrix(self, camera_i, root='calibrations'):
        intrinsic_camera_path = os.path.join(root, 'intrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                        intrinsic_camera_matrix_filenames[camera_i]),
                                            flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
        fp_calibration.release()

        extrinsic_camera_path = os.path.join(root, 'extrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                        extrinsic_camera_matrix_filenames[camera_i]),
                                            flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
        R_z = fp_calibration.getNode('R_z').real()
        fp_calibration.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix, R_z
