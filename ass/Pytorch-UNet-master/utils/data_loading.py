import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))

    # core = idx.replace('RGB_', '')               # 'ar037_2019_n_06_04_0'
    # mask_name = f"mask_{core}{mask_suffix}"      # 'mask_ar037_2019_n_06_04_0'
    # mask_file = list(mask_dir.glob(mask_name + '.*'))[0]
    # mask = np.asarray(load_image(mask_file))

    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

def compute_ndvi(rgb: np.ndarray, nrg: np.ndarray, eps=1e-6):
    # nrg[0] = NIR, nrg[1] = R
    nir = nrg[0].astype(np.float32)
    red = nrg[1].astype(np.float32)
    ndvi = (nir - red) / (nir + red + eps)
    # 归一化到 [0,1]
    ndvi = (ndvi + 1) / 2
    return ndvi[np.newaxis, ...]

def spectral_scale(img: np.ndarray, scale_range=(0.9, 1.1)):
    """
    img: numpy array, shape (C, H, W), float in [0,1]
    scale_range: 放缩因子的取值范围
    """
    C, H, W = img.shape
    for c in range(C):
        factor = np.random.uniform(*scale_range)
        img[c] = np.clip(img[c] * factor, 0.0, 1.0)
    return img

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, dir_nrg: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.augment = False
        self.images_dir = Path(images_dir)
        self.nrg_dir = Path(dir_nrg)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))
        # unique = []
        # for _id in tqdm(self.ids, total=len(self.ids)):
        #     um = unique_mask_values(
        #         _id,
        #         mask_dir=self.mask_dir,
        #         mask_suffix=self.mask_suffix
        #     )
        #     unique.append(um)
        #
        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        self.mask_values = [1]
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            # mask = np.zeros((newH, newW), dtype=np.int64)
            # for i, v in enumerate(mask_values):
            #     if img.ndim == 2:
            #         mask[img == v] = i
            #     else:
            #         mask[(img == v).all(-1)] = i
            # return mask
            return (img > 0).astype(np.int8)

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        nrg_file = list(self.nrg_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        nrg = load_image(nrg_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        nrg = self.preprocess(self.mask_values, nrg, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        ndvi = compute_ndvi(img, nrg)
        r_chan = img[0:1, ...]  # (1, H, W)
        g_chan = img[1:2, ...]  # (1, H, W)
        b_chan = img[2:3, ...]  # (1, H, W)
        nir_chan = nrg[0:1, ...]  # (1, H, W)

        # use = [img, nrg, ndvi]
        # use = [r_chan, g_chan, nir_chan, ndvi]
        use = [r_chan, g_chan, b_chan, nir_chan, ndvi]
        # use = [r_chan, g_chan, b_chan, nir_chan]
        # use = [g_chan,ndvi]
        # [r, g, b, n, ndvi]-2.5bei

        img = np.vstack(use)

        if self.augment:
            img = spectral_scale(img, scale_range=(0.9, 1.1))
            noise = np.random.normal(0, 0.02, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, dir_nrg, mask_dir, scale=1):
        super().__init__(images_dir, dir_nrg, mask_dir, scale, mask_suffix='_mask')
