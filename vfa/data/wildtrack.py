import os, json, re, sys
sys.path.append(os.getcwd())
import numpy as np
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.datasets import VisionDataset
from scipy.sparse import coo_matrix
from vfa.utils import Obj2D 
from vfa.data.GK import GaussianKernel

intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']

WILDTRACK_BBOX_LABEL_NAMES = ['Person']

class Wildtrack(VisionDataset):
    grid_reduce = 4
    img_reduce = 4
    def __init__(self, root, # The Path of MultiviewX
                       world_size = [480, 1440],
                       img_size = [1080, 1920],
                       cube_LWH = [4, 4, 4], # need to scaled!
                       force_download=False, 
                       reload_GK=True):
        super().__init__(root)
        # WILDTRACK has ij-indexing: H*W=480*1440, so x should be \in [0,480), y \in [0,1440)
        # WILDTRACK has in-consistent unit: centi-meter (cm) for calibration & pos annotation
        self.__name__ = 'Wildtrack'
        self.num_cam, self.num_frame = 7, 2000
        self.img_shape, self.world_size, self.cube_LWH = img_size, world_size, cube_LWH # H,W; N_row,N_col; h w l of cube
        self.grid_reduce, self.img_reduce = Wildtrack.grid_reduce, Wildtrack.img_reduce 
        self.reduced_grid_size = list(map(lambda x: int(x / self.grid_reduce), self.world_size)) # 120, 360
        self.reload_GK = reload_GK
        self.label_names = WILDTRACK_BBOX_LABEL_NAMES
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

        self.GK = GaussianKernel(save_dir=r'vfa/data/wt_GK.npy')
        # different from 3D detection task, we only focus on the location detection on MultiviewX 
        # # different from 3D detection task. Thus, `classAverage` is None by default.
        self.classAverage = None 
        self.labels, self.heatmaps = self.download()

        # Create gt.txt file to evaluate MODA, MODP, prec, rcll metrics
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(1, self.num_cam+1)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) 
            if cam >= self.num_cam+1:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths
    
    @staticmethod
    def get_worldgrid_from_pos(pos):
        grid_x = pos % 480
        grid_y = pos // 480
        return np.array([grid_x, grid_y], dtype=int)
    
    @staticmethod
    def get_pos_from_worldgrid(worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 480

    @staticmethod
    def get_worldgrid_from_worldcoord(world_coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        coord_x, coord_y = world_coord
        grid_x = (coord_x + 300) / 2.5
        grid_y = (coord_y + 900) / 2.5
        return np.array([grid_x, grid_y], dtype=int)
    
    @staticmethod
    def get_worldcoord_from_worldgrid(worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        dim = worldgrid.shape[0]
        if dim == 2:
            grid_x, grid_y = worldgrid
            coord_x = -300 + 2.5 * grid_x
            coord_y = -900 + 2.5 * grid_y
            return np.array([coord_x, coord_y])
        elif dim == 3:
            grid_x, grid_y, grid_z = worldgrid
            coord_x = -300 + 2.5 * grid_x
            coord_y = -900 + 2.5 * grid_y
            coord_z = 2.5 * grid_z
            return np.array([coord_x, coord_y, coord_z])


    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 1920 - 1), min(bottom, 1080 - 1)]
        return bbox_by_pos_cam

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    # TODO: check: xy ? yx? Done, yx
    def download(self):

        labels = list()
        # if GK not exist (true), build GK; else, load GK from file. (GK: gaussian kernel heatmap)
        BuildGK = self.reload_GK or not self.GK.GKExist() 
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            # frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            i_s, j_s, v_s = [], [], []
            man_infos = list()
            
            for single_pedestrian in all_pedestrians:
                x, y = self.get_worldgrid_from_pos(single_pedestrian['positionID'])
                location = np.array([x, y, np.zeros_like(x, dtype=x.dtype)]) # NOTICE: different from MultiviewX
                man_infos.append(Obj2D(classname='Person', location=location, conf=None))

                if BuildGK:
                    i_s.append(int(x / self.grid_reduce))
                    j_s.append(int(y / self.grid_reduce))
                    v_s.append(1)
            if BuildGK:
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reduced_grid_size)
                self.GK.add_item(occupancy_map.toarray())

            labels.append(man_infos)
            
        if BuildGK:
            # dump RGK to file
            heatmaps = self.GK.dump_to_file()
        else:
            heatmaps = self.GK.load_from_file()
        
        return labels, heatmaps
                
    
    