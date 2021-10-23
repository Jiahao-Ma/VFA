#TODO: define basic information of MultiviewC
#TODO: define annotation path, image path and calibration matrix
import numpy as np
from PIL import Image
import json, os, sys, cv2
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm
sys.path.append(os.getcwd())

from moft.data.GK import RotationGaussianKernel
from moft.data.ClsAvg import ClassAverage
from moft.utils import Obj3D

intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml', 'intr_Camera7.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml', 'extr_Camera7.xml']

MULTIVIEWC_BBOX_LABEL_NAMES = ['Cow']

class MultiviewC(VisionDataset):
    def __init__(self, root, # PATH of MultiviewC dataset
                       ann_root=r'annotations', 
                       img_root =r'images', 
                       calib_root=r'calibrations', 
                       world_size= [3900, 3900],
                       img_shape = [720, 1280],
                       cube_LWH =  [25, 25, 32],
                       reload_RGK=False
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
        self.img_shape, self.world_size = img_shape, world_size # H, W, N_row, N_col
        self.cube_LWH = cube_LWH # the size of volume of 3D grid, length, width and height
        self.reduced_grid_size = (np.array(self.world_size) // np.array(self.cube_LWH[:2])).astype(np.int32).tolist()
        self.num_cam, self.num_frame = 7, 560
        self.ann_root = os.path.join(root, ann_root)
        self.img_root = os.path.join(root, img_root)
        self.calib_root = os.path.join(root, calib_root)
        self.label_names = MULTIVIEWC_BBOX_LABEL_NAMES
        self.intrinsic_matrices, self.extrinsic_matrices, self.R_z = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam, root=self.calib_root) for cam in range(self.num_cam)])

        self.RGK = RotationGaussianKernel()
        self.reload_RGK=reload_RGK
        self.classAverage = ClassAverage(classes=['Cow'])
        self.labels, self.heatmaps = self.download()

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
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix, R_z

    def download(self):
        ann_paths = [ os.path.join(self.ann_root, p) for p in sorted(os.listdir(self.ann_root)) ]
        labels = list()
        # if cls avg not exist (true), calculate the property of dataset, including mean, number and sum of dimension
        BuildClsAvg = not os.path.exists(self.classAverage.save_path) 
        # if RGK not exist (true), build RGK; else, load RGK from file
        BuildRGK = self.reload_RGK or not self.RGK.RGKExist() 
        with tqdm(total=len(ann_paths), postfix=dict, mininterval=0.3) as pbar:
            for i, ann_path in enumerate(ann_paths):
                with open(ann_path, 'r') as f:
                    annotations = json.load(f)
                cow_infos = list()
                heatmap = np.zeros(self.reduced_grid_size, dtype=np.float32)
                for cows in annotations['C1']:
                    location = cows['location']
                    dimension = cows['dimension']
                    rotation = np.deg2rad(cows['rotation']) # -180~180 => -pi~pi 
                    cow_infos.append(Obj3D(classname='Cow', dimension=dimension, 
                                        location=location, rotation=rotation, conf=None))
                    if BuildRGK:
                        x, y, _ = location
                        _, w, l = dimension
                        box_cx = x * self.reduced_grid_size[0] / self.world_size[0]
                        box_cy = y * self.reduced_grid_size[1] / self.world_size[1]
                        heatmap = self.RGK.gaussian_kernel_heatmap(heatmap, box_cx, box_cy, l, w, cows['rotation'])
                    if BuildClsAvg:
                        self.classAverage.add_item('Cow', dimension)
                if BuildRGK:
                    self.RGK.add_item(heatmap)
                labels.append(cow_infos)
                pbar.set_postfix(**{ 'Process' : ' {} / {}'.format(i, len(ann_paths)),
                                    'Fname' : ' {}'.format(os.path.basename(ann_path)) })
                pbar.update(1)

        if BuildClsAvg:
            self.classAverage.dump_to_file()
        else:
            self.classAverage.load_from_file()
        if BuildRGK:
            # dump RGK to file
            heatmaps = self.RGK.dump_to_file()
        else:
            heatmaps = self.RGK.load_from_file()

        return labels, heatmaps