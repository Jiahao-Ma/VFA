import numpy as np
import os


class RotationGaussianKernel(object):
    # rotated gaussian kernel
    def __init__(self, save_dir=r'vfa/data/RGK.npy',
                 alpha=0.01,
                 GKRatio=8):
        self.save_dir = save_dir
        self.GKRatio = GKRatio
        self.alpha = alpha
        self.heatmaps = list()

    
    def gaussian_kernel_heatmap(self, heatmap, box_cx, box_cy, box_l, box_w, angle, alpha=0.01, gaussian_kernel_ratio=8):
        """
            Args:
                heatmap: the 128 x 128 grids storing the location of target
                box_cx, box_cy: the location of target in heatmap (bottom center of 3D boxes in BEV)
                box_l: the length of 3D box corresponding to the x axis
                box_w: the width of 3D box corresponding to the y axis
                angle: global ration angle
                alpha: determine the gaussian kernel size of target in 128x128 heatmap, larger alpha, larger gaussian kernel will have.
                gaussian_kernel_ratio: also determine the size of gaussian kernel. Below comment has more details.
        """
        heatmap_dtype = heatmap.dtype
        std_w = box_w * alpha # y
        std_l = box_l * alpha # x
        var_w = std_w ** 2
        var_l = std_l ** 2
        # Create a gaussian kernel whose size depends on the maximum side of box's width and length
        # Also, this gaussian kernel is used to store the distribution of target. Because the target is almost 
        # rectangle in MultiviewC, the distribution of gaussian kernel will be elliptical.
        kernel_size = np.int(np.ceil(np.maximum(std_w, std_l)) * gaussian_kernel_ratio)

        xx, yy = np.meshgrid(np.arange(-kernel_size//2, kernel_size//2 + 1, dtype=heatmap_dtype), np.arange(-kernel_size//2, kernel_size//2 + 1, dtype=heatmap_dtype))
            
        gaussian_kernel = np.exp( - (xx)**2 / (2. * var_l) - (yy)**2 / (2. * var_w))
        
        gaussian_kernel = self.bi_rotate(gaussian_kernel, angle)
        # In order to ensure that there is a unique center, firstly find the center of gaussian kernel. 
        # In order to ensure that there is a unique center, the center of the Gaussian kernel is used
        # as the origin for assignment. 
        gaussian_center = np.where(gaussian_kernel == gaussian_kernel.max())
        g_l = gaussian_center[1].item() 
        g_r = gaussian_kernel.shape[1] - gaussian_center[1].item()
        g_t = gaussian_center[0].item() 
        g_b = gaussian_kernel.shape[0] - gaussian_center[0].item()
        # Padding operation can ensure the rotated gaussian kernel does not exceed the boundary of heatmap.
        pad = kernel_size//2
        heatmap = np.pad(heatmap, pad_width=[[pad, pad],[pad, pad]], mode='constant', constant_values=0)
        padded_cx = box_cx + pad 
        padded_cy = box_cy + pad 
        l = int(padded_cx) - g_l
        r = int(padded_cx) + g_r
        t = int(padded_cy) - g_t
        b = int(padded_cy) + g_b
        heatmap[t:b, l:r] = np.maximum(heatmap[t:b, l:r], gaussian_kernel)
        heatmap = heatmap[pad:-pad, pad:-pad]

        # Determine the unique center of gaussian kernel that also represents the valid and correct location of target.
        heatmap[int(box_cy), int(box_cx)] = 1
        return heatmap
    
    def bi_rotate(self, Array, angle, rotate_mode='Clockwise'):
        """
            Bilinear interpolation when rotating
        """
        assert rotate_mode in ['Clockwise', 'Counterclockwise'], 'Rotate Mode Error.'
        H, W = Array.shape

        pi = np.pi
    
        angle = angle * pi / 180

        matrix1 = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [-0.5 * H, 0.5 * W, 1]])
        if rotate_mode == 'Counterclockwise':
        # counter clockwise
            matrix2 = np.array([[np.cos(angle), np.sin(angle), 0],
                                [-np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
        elif rotate_mode == 'Clockwise':
            matrix2 = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])

        matrix3 = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0.5 * H, 0.5 * W, 1]])

        new_data = np.zeros_like(Array,dtype=Array.dtype)

        for i in range(H):
            for j in range(W):

                dot1 = np.matmul(np.array([i, j, 1]), matrix1)
                dot2 = np.matmul(dot1, matrix2)
                dot3 = np.matmul(dot2, matrix3)

                new_coordinate = dot3

                new_i = int(np.floor(new_coordinate[0]))
                new_j = int(np.floor(new_coordinate[1]))

                u = new_coordinate[0] - new_i
                v = new_coordinate[1] - new_j

                if new_j>=W or new_i >=H or new_i<1 or new_j<1 or (i+1)>=H or (j+1)>=W:
                    continue

                if (new_i + 1)>=H or (new_j+1)>=W:
                    new_data[i, j] = Array[new_i,new_j]

                else:
                    new_data[i, j] = (1-u)*(1-v)*Array[new_i,new_j] + \
                                    (1-u)*v*Array[new_i,new_j+1] + \
                                    u*(1-v)*Array[new_i+1,new_j] +\
                                    u*v*Array[new_i+1,new_j+1]
        return new_data
    
    def add_item(self, heatmap):
        self.heatmaps.append(heatmap)

    def RGKExist(self):
        return os.path.exists(self.save_dir)
    
    def load_from_file(self):
        try:
            return np.load(self.save_dir)
        except:
            print('\033[31mRGK load error.\033[0m ')
    
    def dump_to_file(self):
        if isinstance(self.heatmaps, list):
            self.heatmaps = np.stack(self.heatmaps, axis=0)
        try:
            np.save(self.save_dir, self.heatmaps)
            print('RGK has been save %s' %self.save_dir)
        except:
            print('\033[31mRGK save error.\033[0m')
        return self.heatmaps
