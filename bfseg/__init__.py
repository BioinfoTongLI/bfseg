import numba
import numpy as np
from . import watershed, findmax
from scipy.ndimage import label, distance_transform_edt
from skimage.measure import regionprops
from tifffile.tifffile import TiffFile
from skimage.filters import threshold_otsu
import os


@numba.jit(nopython=True, parallel=True)
def find_focus(chunk):
    d, h, w = chunk.shape
    min_var = np.inf
    min_ind = int(d/2)
    for i in range(d):
        tmp_v = np.std(chunk[i, :, :])
        if tmp_v < min_var:
            min_var = tmp_v
            min_ind = i
    return min_ind


@numba.jit(nopython=True, parallel=True)
def get_chunk_mask(h, w, M, N):
    if h%M !=0 or w%N !=0:
        print("image can not be evenly divided")
    ch_h, ch_w = int(h/M), int(w/N)
    mask = np.array([1] * h * w).reshape(h, w)
    labels = list(range(M * N))
    count = 0
    for i in range(M):
        for j in range(N):
            mask[i*ch_h : (i+1) * ch_h,
                 j*ch_w : (j+1) * ch_w] = labels[count]
            count += 1
    return mask


@numba.jit(nopython=False, parallel=True)
def get_focus_grid(img, mask):
    d, h, w = img.shape
    n_chunk = mask.max()
    focus_img = mask.copy()
    for ind in range(n_chunk + 1):
        min_var = np.inf
        min_ind = int(d/2)
        cur_mask = mask == ind
        for i in range(d):
            # boolean slicing is not supported yet... so go python mode
            tmp_v = img[i, :, :][cur_mask].std()
            if tmp_v < min_var:
                min_var = tmp_v
                min_ind = i
        focus_img[cur_mask] = min_ind
    return focus_img


@numba.jit(nopython=True, parallel=True)
def preCalculateParameters(first_ind=0, last_ind=33, N=101, direction=-1, zf=17,
        sigma=8.0):
    h = (last_ind - first_ind) / N
    zs_d = np.array([first_ind + h * i for i in range(N)])
    zs_i = [int(i) for i in zs_d]
    return h, zs_i, [direction * (zs_d[i] - zf) * np.exp(-(zf - zs_d[i]) ** 2 / (2 * (sigma ** 2))) for i in range(N)]


@numba.jit(nopython=True, parallel=True)
def integrate(z_pile, h, zs_i, smooth_ponderation, first_ind=0, last_ind=33, N=101):
    total = z_pile[zs_i[first_ind]] * smooth_ponderation[first_ind] + z_pile[zs_i[N - 1]] * smooth_ponderation[N - 1]
    for i in range(1, N, 2):
        total += 4.0 * z_pile[zs_i[first_ind + int(i)]] * smooth_ponderation[first_ind + int(i)]
    for i in range(2, N-1, 2):
        total += 2.0 * z_pile[zs_i[first_ind + int(i)]] * smooth_ponderation[first_ind + int(i)]
    return total * h / 3.0;


@numba.jit(nopython=True, parallel=True)
def compute(img, integrated, zf, focus_img, h, zs_i, smoothed_ponderation):
    depth, height, width = img.shape
    for x in range(height):
        for y in range(width):
            if focus_img[x, y] == zf:
                integrated[x, y] = integrate(img[:, x, y], h, zs_i,
                        smoothed_ponderation, last_ind=depth)


@numba.jit(parallel=True)
def segment(bf_stack, region_mask, zf_params):
    focus_img = get_focus_grid(bf_stack, region_mask)
    h, w = focus_img.shape
    for zf in np.unique(focus_img):
        if zf not in zf_params:
            zf_params[zf] = preCalculateParameters(zf=zf, last_ind=bf_stack.shape[0])
    integrated = np.zeros(h * w).reshape(h, w)
    for zf in np.unique(focus_img):
        compute(bf_stack, integrated, zf, focus_img, *zf_params[zf])
    th_otsu = threshold_otsu(integrated)
    return integrated, integrated <= th_otsu, zf_params


def get_master_fhs(root, pref):
    for r, ds, fs in os.walk(root):
        for f in fs:
            if f.endswith(".tif") and f.startswith(pref) and\
                not f.startswith(pref + "_end") and not f.endswith("_1.ome.tif"):
                img_path = r + os.sep + f
                with TiffFile(img_path) as imgs:
                    if len(imgs.series) > 1:
                        yield r, imgs


def ipy_watershed(img, tol):
    dist = -distance_transform_edt(img)
    pts = findmax.find_maximum(dist, tol, False)
    buf = np.zeros(img.shape, dtype=np.uint16)
    buf[pts[:,0], pts[:,1]] = 1
    markers, n = label(buf, np.ones((3,3)))
    line = watershed.watershed(dist, markers, line=True, conn=False+1)
    img[line==0] = 0
    return img
