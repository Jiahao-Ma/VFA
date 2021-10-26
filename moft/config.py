from moft.data.wildtrack import Wildtrack
from moft.utils import to_numpy
from moft.data.multiviewX import MultiviewX

class MultiviewC_Config(object):
    name = 'MultiviewC'

    mode = '3D'

    root = r'~\Data\MultiviewC' # the Path of MultiviewC dataset

    world_size = (3900, 3900) # width and length of designed grid

    image_size = (720, 1280)
    
    resize_size = (720, 1280)

    ann = r'annotations'
    
    calib = r'calibrations'

    grid_h = 160 # the height of designed grid

    cube_size = to_numpy((25, 25, 32))

    loss_weight = [1., 1., 1., 1.]

    grid_scale = 1. # make the ratio and scale of grid correspond, which also project the design voxel to image successfully.



class MultiviewX_Config(object):
    name = 'MultiviewX'

    mode = '2D'

    root = r'~\Data\MultiviewX' # the Path of MultiviewX dataset

    world_size = (640, 1000) # width and length of designed grid

    image_size =  (1080, 1920) # (1080, 1920) -> (720, 1280) to reduce memory

    resize_size = (1080, 1920)

    ann = r'annotations_positions'
    
    calib = r'calibrations'

    grid_h = 72 # the height of designed grid 
    # case1: 4; case2: 16; case3: 32; case4: 48; case5: 64; case6: 72; case7: 80

    cube_size = to_numpy((4, 4, 8))
    # case1: 4; case2: 4; case3: 4; case4:8; case5: 8; case6: 8; case7: 16

    loss_weight = [5., 1.]

    grid_scale = 40.  # make the ratio and scale of grid correspond, which also project the design voxel to image successfully.


class Wildtrack_Config(object):
    name = 'Wildtrack'

    mode = '2D'

    root = r'~\Data\Wildtrack' # the Path of MultiviewX dataset

    world_size = (480, 1440) # width and length of designed grid

    image_size =  (1080, 1920) # (1080, 1920) -> (720, 1280) to reduce memory

    resize_size = (1080, 1920)

    ann = r'annotations_positions'
    
    calib = r'calibrations'

    grid_h = 64 # the height of designed grid 
    

    cube_size = to_numpy((4, 4, 8))
    

    loss_weight = [5., 1.]

    grid_scale = 1.  # make the ratio and scale of grid correspond, which also project the design voxel to image successfully.


mc_opts = MultiviewC_Config()
mx_opts = MultiviewX_Config()
wt_opts = Wildtrack_Config()