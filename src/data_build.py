
# boring imports
import os, pathlib
import logging
import h5py
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.ndimage, scipy.spatial
import numpy as np

from matplotlib import cm

logger = logging.getLogger("dataset - building")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
fh = logging.FileHandler("count.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s %(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query 4 nearest neighbors
    distances, locations = tree.query(pts, k=4)

    logger.info('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    logger.info('done.')
    return density


root = pathlib.Path("./ShanghaiTech")
part_A = root/"part_A/"
part_B = root/"part_B/"

def compute_density_ST(part):
    """compute density for ShanghaiTech dataset
    """
    # create destination folders
    dest_folder_test = part/"test_data/densities"
    dest_folder_train = part/"train_data/densities"
    dest_folder_test.mkdir(exist_ok=True)
    dest_folder_train.mkdir(exist_ok=True)

    # create lists of necessary paths
    img_paths = [str(path) for path in part.rglob("*.jpg")]
    gt_paths = [str(path).replace('images','ground-truth').replace('IMG_','GT_IMG_').replace('.jpg','.mat') for path in img_paths]
    dens_paths = [str(path).replace('images','densities').replace('IMG_','DENS_').replace('.jpg','.npy') for path in img_paths]
    paths = list(zip(img_paths, gt_paths, dens_paths))

    for img, gt, dens in paths:
        mat = io.loadmat(gt)
        logger.debug(f"image: {img}")
        img = plt.imread(img)
        k = np.zeros((img.shape[0],img.shape[1]))
        gt = mat["image_info"][0,0][0,0][0]
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        k = np.asarray(gaussian_filter_density(k))
        logger.debug(f"density shape: {k.shape}")
        logger.debug(f"density count: {np.ceil(np.sum(k))}")
        logger.debug(f"actual count: {len(gt)}")
        np.save(dens, k)



if __name__ == "__main__":
    compute_density_ST(part_A)

