import torch
import matplotlib.pyplot as plt
import os, sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.getcwd())
from moft.data.wildtrack import Wildtrack
from moft.data.dataset import frameDataset, MultiviewC
from moft.visualization.bbox import *

def get_worldcoord_from_imagecoord(image_coord, project_mat):
    # uv = P @ XYZ -> P` @ uv = XYZ
    # image_coord size: [2, N] -> [3, N]
    image_coord = np.concatenate([image_coord, np.ones(shape=(1, image_coord.shape[1]))], axis=0)
    # project_mat size: [3, 4] -> [3, 3]
    project_mat = np.linalg.inv(np.delete(project_mat, 2, 1))
    # permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    # project_mat = permutation_mat @ project_mat
    world_coord = project_mat @ image_coord
    world_coord = world_coord[:2, :] / world_coord[2, :]
    return world_coord

def get_imagecoord_from_worldcoord(world_coord, project_mat):
    project_mat = np.delete(project_mat, 2, 1)
    world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
    image_coord = project_mat @ world_coord
    image_coord = image_coord[:2, :] / image_coord[2, :]
    return image_coord

def project(world_coord, calib):
    # (3, 4) (4, N)
    if world_coord.shape[0] != 4:
        world_coord = world_coord.T
    img_coord = calib @ world_coord
    img_coord = img_coord[:2, :] / img_coord[2, :]
    return img_coord.T

def make_grid(world_size=(3900, 3900), grid_offset=(0, 0, 0), cube_LW=[25, 25]):
    """
        *********
        *       *
        *       * y
        *********
                x
    """
    length, width = world_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0., width, cube_LW[0]) + xoff
    ycoords = torch.arange(0., length, cube_LW[1]) + yoff
    
    yy, xx = torch.meshgrid(ycoords, xcoords)
    return torch.stack([xx, yy, torch.full_like(xx, zoff)], dim=-1)
    

if __name__ == '__main__':
    from moft.data.multiviewX import MultiviewX
    from moft.visualization.figure import visualize_image
    mx = MultiviewX(root=r'F:\ANU\ENGN8602\Data\MultiviewX')
    wt = Wildtrack(root=r'F:\ANU\ENGN8602\Data\Wildtrack')
    # plane_shape = np.array(mx.reduced_grid_size)
    # xx, yy = np.meshgrid(np.arange(0, 1920, 30), np.arange(0, 1080, 30))
    # grid2d = np.stack([xx, yy], axis=2).reshape(-1, 2)

    # data = frameDataset(Wildtrack(root=r'F:\ANU\ENGN8602\Data\MultiviewX'))
    data = frameDataset(Wildtrack(root=r'F:\ANU\ENGN8602\Data\Wildtrack'))
    index, images, objects, heatmaps, calibs, grid  = next(iter(data))
    # for cam in range(0, mx.num_cam):
    #     image = visualize_image(images[cam]).transpose(1, 2, 0)
    #     plt.subplot(131)
    #     plt.imshow(image)
    #     plt.scatter(grid2d[:, 0], grid2d[:, 1], s=1, c='red')
    #     # bottom2d = project(bottom3d, calibs[cam])
    #     # plt.scatter(bottom2d[:, 0], bottom2d[:, 1], s=1, c='red')
    #     plt.xlim(0, 1920)
    #     plt.ylim(1080, 0)

    #     plt.subplot(132)
    #     plane = np.zeros(shape=(640, 1000))
    #     grid3d = get_worldcoord_from_imagecoord(grid2d.T, calibs[cam])
        # grid3d = MultiviewX.get_worldgrid_from_worldcoord(grid3d).T 
    #     # grid3d = np.array(grid3d, dtype=np.int32).T

    #     inside_grid = list()
    #     for xy in grid3d:
    #         x, y = xy
    #         if x in range(0, 1000) and y in range(0, 640):
    #             plane[y, x] += 1
    #             inside_grid.append(xy[None, :])

    #     inside_grid = np.concatenate(inside_grid, axis=0)
    #     plane = (plane * 255).clip(0, 255).astype(np.uint8)
    #     plt.imshow(plane)

    #     plt.subplot(133)
    #     plt.imshow(image)
    #     worldcoord = MultiviewX.get_worldcoord_from_worldgrid(inside_grid.T)
    #     imgcoord = get_imagecoord_from_worldcoord(worldcoord, calibs[cam]).T
    #     plt.scatter(imgcoord[:, 0], imgcoord[:, 1], c='green', s=3)
    #     plt.xlim(0, 1920)
    #     plt.ylim(1080, 0)
    #     plt.show()
    # r=1
    # plt.figure(figsize=(15, 8))
    # xx, yy = np.meshgrid(np.arange(0, 1000, 4), np.arange(0, 640, 4))
    # worldgrid = np.stack([xx, yy], axis=2).reshape(-1, 2)
    # worldgrid = np.concatenate([worldgrid, np.ones((worldgrid.shape[0], 1))*0, np.ones((worldgrid.shape[0], 1))], axis=1) 
    # for cam in range(0, mx.num_cam):
    #     plt.subplot(131)
    #     plane = np.zeros((640, 1000))
    #     for pts in worldgrid:
    #         pts = pts[:2].astype(np.int32)
    #         plane[pts[1]-r:pts[1]+r, pts[0]-r:pts[0]+r] += 1
    #     plane = (plane * 255).astype(np.uint8)
    #     plt.imshow(plane)

    #     # worldcoord = MultiviewX.get_worldcoord_from_worldgrid(worldgrid.transpose())
    #     worldcoord = worldgrid / np.array([40, 40, 40, 1])
    #     # imgcoord = get_imagecoord_from_worldcoord(worldcoord, calibs[cam]).T
    #     imgcoord = project(worldcoord, calibs[cam])
    #     image = visualize_image(images[cam]).transpose(1, 2, 0)
    #     plt.subplot(132)
    #     plt.imshow(image)
    #     imgcoord = imgcoord.astype(np.int32)
    #     mask = (imgcoord[:, 0]>0) * (imgcoord[:, 0]<1920) * (imgcoord[:, 1]>0) * (imgcoord[:, 1]<1080)
    #     imgcoord = imgcoord[mask]
    #     plt.scatter(imgcoord[:, 0], imgcoord[:, 1], c='red', s=5)

    #     plt.subplot(133)
    #     plane = np.zeros((160, 250))
    #     for pts in worldgrid[mask]:
    #         pts = (pts / np.array([1000, 640, 1, 1]) * np.array([250, 160, 1, 1])).astype(np.int32)
    #         pts = pts[:2]
    #         plane[pts[1]-r:pts[1]+r, pts[0]-r:pts[0]+r] += 1
    #     plane = (plane * 255).astype(np.uint8)
    #     plt.imshow(plane)

    #     plt.show()
    r=1
    plt.figure(figsize=(15, 8))
    xx, yy = np.meshgrid(np.arange(0, 480, 40), np.arange(0, 1440, 40))
    worldgrid = np.stack([xx, yy], axis=2).reshape(-1, 2)
    worldgrid = np.concatenate([worldgrid, np.ones((worldgrid.shape[0], 1))*0], axis=1)  #, np.ones((worldgrid.shape[0], 1))
    for cam in range(0, mx.num_cam):
        plt.subplot(131)
        plane = np.zeros((480, 1440))
        for pts in worldgrid:
            pts = pts[:2].astype(np.int32)
            plane[pts[0]-r:pts[0]+r, pts[1]-r:pts[1]+r] += 1
        plane = (plane * 255).astype(np.uint8)
        plt.imshow(plane)

        # worldcoord = MultiviewX.get_worldcoord_from_worldgrid(worldgrid.transpose())
        worldcoord = Wildtrack.get_worldcoord_from_worldgrid(worldgrid.T).T
        worldcoord = np.concatenate([worldcoord, np.ones((worldcoord.shape[0], 1))], axis=1)
        # imgcoord = get_imagecoord_from_worldcoord(worldcoord, calibs[cam]).T
        imgcoord = project(worldcoord, calibs[cam])
        image = visualize_image(images[cam]).transpose(1, 2, 0)
        plt.subplot(132)
        plt.imshow(image)
        imgcoord = imgcoord.astype(np.int32)
        mask = (imgcoord[:, 0]>0) * (imgcoord[:, 0]<1920) * (imgcoord[:, 1]>0) * (imgcoord[:, 1]<1080)
        imgcoord = imgcoord[mask]
        plt.scatter(imgcoord[:, 0], imgcoord[:, 1], c='red', s=5)

        plt.subplot(133)
        plane = np.zeros((120, 360))
        for pts in worldgrid[mask]:
            pts = (pts / np.array([480, 1440, 1]) * np.array([120, 360, 1])).astype(np.int32)
            pts = pts[:2]
            plane[pts[0]-r:pts[0]+r, pts[1]-r:pts[1]+r] += 1
        plane = (plane * 255).astype(np.uint8)
        plt.imshow(plane)

        plt.show()