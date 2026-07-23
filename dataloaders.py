import os
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import random
from tqdm import tqdm
import albumentations as A
import cv2

# Dataset statistics for normalization
# These values are pre-computed for the dataset with background subtraction
sample_mu = [-1.2223, -1.8114, -1.7090]
sample_std = [11.7932, 12.7956, 13.6452]

# Statistics for depth maps
dmap_mu = [8.8114]
dmap_std = [30.3624]

# Statistics for surface normals
norm_mu = [126.9855, 127.2061, 247.7740]
norm_std = [25.1953, 25.5532, 22.1101]

# Statistics for raw (non-background-subtracted) tactile images
# Simple [-1, 1] normalization; recompute with __main__ for exact values
raw_mu = [127.5, 127.5, 127.5]
raw_std = [127.5, 127.5, 127.5]

# ImageNet normalization stats scaled to [0, 255] float32 input
# (standard [0.485,0.456,0.406] / [0.229,0.224,0.225] × 255)
imagenet_mu = [123.675, 116.28, 103.53]
imagenet_std = [58.395, 57.12, 57.375]

import math

FIXED_CROP = 1.0 / math.sqrt(2.0)


def fixed_center_crop(img, out_size=None):
    """Center-crop to 1/sqrt(2) of the side, resize back to original (or out_size).
    Applied to EVERY sample (train+val) so rotations up to 45° have no border artifacts."""
    H, W = img.shape[:2]
    side = int(math.floor(min(H, W) * FIXED_CROP))
    off_y, off_x = (H - side) // 2, (W - side) // 2
    crop = img[off_y:off_y + side, off_x:off_x + side]
    out = out_size or (W, H)
    if isinstance(out, int):
        out = (out, out)
    return cv2.resize(crop, out, interpolation=cv2.INTER_LINEAR)


def depth_to_normal(depth, pixel_size_x, pixel_size_y):
    """Compute unit surface normals from a depth map via central finite differences."""
    dz_dx = np.zeros_like(depth)
    dz_dy = np.zeros_like(depth)
    dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / (2.0 * pixel_size_x)
    dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / (2.0 * pixel_size_y)
    dz_dx[:, 0] = (depth[:, 1] - depth[:, 0]) / pixel_size_x
    dz_dx[:, -1] = (depth[:, -1] - depth[:, -2]) / pixel_size_x
    dz_dy[0, :] = (depth[1, :] - depth[0, :]) / pixel_size_y
    dz_dy[-1, :] = (depth[-1, :] - depth[-2, :]) / pixel_size_y
    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1e-8)
    return (normal / norm).astype(np.float32)


def gel_spin_rotate(sample, calib_imgs, depth, normal, angle_deg):
    """Rotate all images by angle_deg around center. Normal vectors are corrected."""
    H, W = sample.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    flags, border = cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101

    sample = cv2.warpAffine(sample, M, (W, H), flags=flags, borderMode=border)
    calib_imgs = [cv2.warpAffine(c, M, (W, H), flags=flags, borderMode=border)
                  for c in calib_imgs]
    if depth is not None:
        depth = cv2.warpAffine(depth, M, (W, H), flags=flags, borderMode=border)
    if normal is not None:
        normal = cv2.warpAffine(normal, M, (W, H), flags=flags, borderMode=border)
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.float32(np.cos(rad)), np.float32(np.sin(rad))
        nx = normal[:, :, 0] / 127.5 - 1.0
        ny = normal[:, :, 1] / 127.5 - 1.0
        normal[:, :, 0] = np.clip((cos_a * nx + sin_a * ny + 1.0) * 127.5, 0, 255)
        normal[:, :, 1] = np.clip((-sin_a * nx + cos_a * ny + 1.0) * 127.5, 0, 255)

    return sample, calib_imgs, depth, normal


DEFAULT_AUGMENT_PARAMS = {
    'gain':    0.5,
    'bias':    45.0,
    'grad':    0.7,
    'bright':  25.0,
    'resid':   20.0,
    'noise':   6.0,
    'rot_deg': 0.0,
    'hflip':   False,
    'vflip':   False,
}


class TactileAugment:
    """Tactile sensor augmentation for DPT training.

    Photometric augmentations (gain, bias, bright, grad, noise, resid)
    are applied to the diff image only.  Geometric augmentations (flip,
    rotate) are applied to diff, calibration diffs, depth AND normal maps
    with proper normal-direction correction.
    """

    def __init__(self, params=None):
        self.p = {**DEFAULT_AUGMENT_PARAMS, **(params or {})}

    def __call__(self, sample_diff, calib_diffs, depth, normal):
        p = self.p
        H, W = sample_diff.shape[:2]

        # ── photometric (sample_diff only) ────────────────────────────
        if p['gain'] > 0:
            g = np.random.uniform(1 - p['gain'], 1 + p['gain'],
                                  size=(1, 1, 3)).astype(np.float32)
            sample_diff = sample_diff * g

        if p['bias'] > 0:
            b = np.random.uniform(-p['bias'], p['bias'],
                                  size=(1, 1, 3)).astype(np.float32)
            sample_diff = sample_diff + b

        if p['bright'] > 0:
            sample_diff = sample_diff + np.float32(
                np.random.uniform(-p['bright'], p['bright']))

        if p['grad'] > 0:
            angle = np.random.uniform(0, 2 * np.pi)
            ys = np.linspace(-1, 1, H, dtype=np.float32).reshape(-1, 1)
            xs = np.linspace(-1, 1, W, dtype=np.float32).reshape(1, -1)
            grad_map = (np.float32(np.cos(angle)) * xs
                        + np.float32(np.sin(angle)) * ys)
            amp = np.random.uniform(0, p['grad'],
                                    size=(1, 1, 3)).astype(np.float32)
            sample_diff = sample_diff + grad_map[..., None] * amp * np.float32(50.0)

        if p['resid'] > 0:
            raw = np.random.randn(16, 16, 3).astype(np.float32)
            smooth = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
            smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=H / 8.0)
            std = np.float32(smooth.std())
            if std > 1e-6:
                smooth = smooth / std * np.float32(p['resid'])
            sample_diff = sample_diff + smooth

        if p['noise'] > 0:
            noise = np.random.normal(0, p['noise'],
                                     sample_diff.shape).astype(np.float32)
            sample_diff = sample_diff + noise

        # ── geometric (all images) ────────────────────────────────────
        do_hflip = p['hflip'] and np.random.random() < 0.5
        do_vflip = p['vflip'] and np.random.random() < 0.5
        rot_angle = 0.0
        if p['rot_deg'] > 0:
            rot_angle = np.random.uniform(-p['rot_deg'], p['rot_deg'])

        if do_hflip:
            sample_diff = np.ascontiguousarray(sample_diff[:, ::-1])
            calib_diffs = [np.ascontiguousarray(c[:, ::-1]) for c in calib_diffs]
            if depth is not None:
                depth = np.ascontiguousarray(depth[:, ::-1])
            if normal is not None:
                normal = np.ascontiguousarray(normal[:, ::-1])
                normal[:, :, 0] = 255.0 - normal[:, :, 0]

        if do_vflip:
            sample_diff = np.ascontiguousarray(sample_diff[::-1])
            calib_diffs = [np.ascontiguousarray(c[::-1]) for c in calib_diffs]
            if depth is not None:
                depth = np.ascontiguousarray(depth[::-1])
            if normal is not None:
                normal = np.ascontiguousarray(normal[::-1])
                normal[:, :, 1] = 255.0 - normal[:, :, 1]

        if abs(rot_angle) > 0.5:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), rot_angle, 1.0)
            flags = cv2.INTER_LINEAR
            border = cv2.BORDER_REFLECT_101

            sample_diff = cv2.warpAffine(sample_diff, M, (W, H),
                                          flags=flags, borderMode=border)
            calib_diffs = [cv2.warpAffine(c, M, (W, H),
                                           flags=flags, borderMode=border)
                           for c in calib_diffs]
            if depth is not None:
                depth = cv2.warpAffine(depth, M, (W, H),
                                       flags=flags, borderMode=border)
            if normal is not None:
                normal = cv2.warpAffine(normal, M, (W, H),
                                         flags=flags, borderMode=border)
                rad = np.radians(rot_angle)
                cos_a = np.float32(np.cos(rad))
                sin_a = np.float32(np.sin(rad))
                nx = normal[:, :, 0] / 127.5 - 1.0
                ny = normal[:, :, 1] / 127.5 - 1.0
                new_nx = cos_a * nx + sin_a * ny
                new_ny = -sin_a * nx + cos_a * ny
                normal[:, :, 0] = np.clip((new_nx + 1.0) * 127.5, 0, 255)
                normal[:, :, 1] = np.clip((new_ny + 1.0) * 127.5, 0, 255)

        return sample_diff, calib_diffs, depth, normal


class sim_dataset(Dataset):
    """
    Custom dataset class for loading and processing sensor data.
    Handles calibration images, samples, depth maps, and surface normals.
    """
    def __init__(self, 
                 path, 
                 augment=False,
                 sendTwo=False, 
                 transforms=T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)]),
                 dmap_transforms=T.Compose([T.ToTensor(), T.Normalize(mean=dmap_mu, std=dmap_std)]),
                 norm_transforms=T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)]),
                 calibration_config=18,
                 num_samples=None,
                 num_sensors=None) -> None:
        # Initialize dataset parameters and paths
        self.path = path
        self.transforms = transforms
        self.dmap_transforms = dmap_transforms
        self.norm_transforms = norm_transforms
        self.sendTwo = sendTwo

        # Configure calibration list based on calibration_config
        if calibration_config == 0: self.calib_list = []
        elif calibration_config == 4: self.calib_list = [1,3,7,9]
        elif calibration_config == 8: self.calib_list = [1,3,7,9,10,12,16,18]
        elif calibration_config == 9: self.calib_list = [i for i in range(1, 10)]
        elif calibration_config == 18: self.calib_list = [i for i in range(1, 19)]
        else: raise ValueError('Invalid calibration configuration')
            
        # Load and sort sensor directories
        sensors = os.listdir(path)
        sensors.sort()
        if '.DS_Store' in sensors: sensors.remove('.DS_Store')
        
        # Get calibration and sample information
        calibrations = os.listdir(osp.join(path, sensors[0], 'calibration'))
        if '.DS_Store' in calibrations: calibrations.remove('.DS_Store')
        
        samples =  os.listdir(osp.join(path, sensors[-1], 'samples'))
        if '.DS_Store' in samples: samples.remove('.DS_Store')
        
        # Set dataset size parameters
        self.num_calibrations = len(calibrations)
        
        if num_samples != None: self.num_samples = num_samples
        else: self.num_samples = len(samples)
        
        if num_sensors != None: self.num_sensors = num_sensors
        else: self.num_sensors = len(sensors)
        
        # Initialize augmentation pipeline
        self.augment = A.Compose([], additional_targets={'c0':'image', 
                                                         'c1':'image', 'c2':'image', 'c3':'image', 
                                                         'c4':'image', 'c5':'image', 'c6':'image', 
                                                         'c7':'image', 'c8':'image', 'c9':'image', 
                                                         'c10':'image', 'c11':'image', 'c12':'image', 
                                                         'c13':'image', 'c14':'image', 'c15':'image', 
                                                         'c16':'image', 'c17':'image', 'c18':'image'})
        if augment == True:
            # Define augmentation transforms for both sample and calibration images
            self.augment = A.Compose([A.ColorJitter(brightness=(0.6, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                                      A.Blur()],
                                      additional_targets={'c0':'image', 
                                                          'c1':'image', 'c2':'image', 'c3':'image', 
                                                          'c4':'image', 'c5':'image', 'c6':'image', 
                                                          'c7':'image', 'c8':'image', 'c9':'image', 
                                                          'c10':'image', 'c11':'image', 'c12':'image', 
                                                          'c13':'image', 'c14':'image', 'c15':'image', 
                                                          'c16':'image', 'c17':'image', 'c18':'image'})

    def __len__(self) -> int:
        return self.num_sensors * self.num_samples
    
    def getitem_helper(self, sensor_idx, sample_idx):
        """
        Helper function to load and process data for a specific sensor and sample.
        Handles background subtraction, calibration data loading, and augmentation.
        """
        # Load background reference image for subtraction
        ref_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '0000.png')
        ref_img = np.array(Image.open(ref_path))
        
        # Load all calibration images
        calib = []
        for i in range(1,19):
            calib_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '{0:04}.png'.format(i))
            calib_img = np.array(Image.open(calib_path))
            calib.append(calib_img)
        
        # Load sample image
        sample_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'samples', '{0:04}.png'.format(sample_idx))
        sample = np.array(Image.open(sample_path)) 
        
        # Apply augmentations to all images
        augments = self.augment(image=sample,
                                c0=ref_img,
                                c1=calib[0], c2=calib[1], c3=calib[2],
                                c4=calib[3], c5=calib[4], c6=calib[5], 
                                c7=calib[6], c8=calib[7], c9=calib[8],
                                c10=calib[9], c11=calib[10], c12=calib[11],
                                c13=calib[12], c14=calib[13], c15=calib[14],
                                c16=calib[15], c17=calib[16], c18=calib[17])
        
        # Process reference image
        ref_img = np.array(augments['c0'], dtype=np.float32)
        
        # Process calibration images with background subtraction
        calib = torch.tensor([])
        for i in self.calib_list:
            calib_img = np.array(augments[f'c{i}'], dtype=np.float32)
            calib_img = augments[f'c{i}'] - ref_img
            calib = torch.cat([calib, self.transforms(calib_img)])
            
        # Process sample image with background subtraction
        sample =  np.array(augments['image'], dtype=np.float32)
        sample = augments['image'] - ref_img
        sample = self.transforms(sample) 
        
        # Load depth map and surface normals if available
        try:
            dmap_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'dmaps', '{0:04}.png'.format(sample_idx))
            dmap = np.array(Image.open(dmap_path), dtype=np.float32)
            dmap = self.dmap_transforms(dmap)

            norm_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'norms', '{0:04}.png'.format(sample_idx))
            norm = np.array(Image.open(norm_path), dtype=np.float32)
            norm = self.norm_transforms(norm)
        except: 
            dmap = None
            norm = None
        
        idx = torch.tensor(sample_idx)
        
        return sample, calib, dmap, norm, idx
    
    def __getitem__(self, index):
        """
        Main data loading function that handles both single and paired data loading.
        """
        if index >= len(self): raise IndexError(f"Index {index} out of range")
        
        # Calculate sensor and sample indices
        sensor_idx = index // self.num_samples
        sample_idx = index % self.num_samples
        
        # Load primary data
        sample, calib, dmap, norm, idx = self.getitem_helper(sensor_idx, sample_idx)
        
        # Handle paired data loading for contrastive learning
        if self.sendTwo: 
            second_sensor_idx = sensor_idx
            while second_sensor_idx == sensor_idx:
                second_sensor_idx = random.randint(0,self.num_sensors-1)
            sample2, calib2, dmap2, norm2, _ = self.getitem_helper(second_sensor_idx, sample_idx)
            
            # Stack paired data
            calib = torch.stack([calib, calib2], dim=0)
            sample = torch.stack([sample, sample2], dim=0) 
            dmap = torch.stack([dmap, dmap2], dim=0)
            norm = torch.stack([norm, norm2], dim=0)
        
        return {'sample': sample, 'calibration': calib, 'dmap': dmap, 'norm' : norm, 'idx': idx}
    
class sim_dataset_nested(Dataset):
    """
    sim_dataset 的巢狀版本，給 gs_blender 產生的資料集用：
        path/<obj>/session_xxx/sensor_0000/{calibration,samples,dmaps,norms}/
    與原本 sim_dataset 的差異：
      1. 目錄是巢狀（<obj>/session_xxx/sensor_*），不是平鋪的 sensor_XXXX。
      2. 每個 unit 只有 1 個 sensor，無法用「同 sample、不同 sensor」做對比正樣本對；
         改用「同一張 sample 的兩次隨機增強」當正樣本對（SimCLR 式）。
    其餘（背景相減、18 張 calibration、normalization、224x224）與原本一致，
    回傳格式也一致：{sample, calibration, dmap, norm, idx}。
    """
    def __init__(self,
                 path,
                 augment=False,
                 sendTwo=False,
                 transforms=T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)]),
                 dmap_transforms=T.Compose([T.ToTensor(), T.Normalize(mean=dmap_mu, std=dmap_std)]),
                 norm_transforms=T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)]),
                 calibration_config=18,
                 num_samples=None,
                 use_gt_norm=False,
                 include_objects=None,        # 只收這些物體的 unit（按物體切 train/val 用）
                 num_sensors=None,            # num_sensors 僅為 API 相容，巢狀版用不到
                 raw_input=False,
                 tactile_augment=False,
                 augment_params=None,
                 gel_spin_max_deg=0.0,
                 center_crop=False,
                 depth_from_npy=False,
                 gel_view_m=0.017502) -> None:
        self.path = path
        self.transforms = transforms
        self.dmap_transforms = dmap_transforms
        self.norm_transforms = norm_transforms
        self.sendTwo = sendTwo
        self.norm_suffix = '_gt' if use_gt_norm else ''
        self._calib_cache = {}   # 每個 unit 的 (ref, 18×calib) 原始影像，避免每 sample 重讀
        self.skip_calibration = False

        if calibration_config == 0: self.calib_list = []
        elif calibration_config == 4: self.calib_list = [1,3,7,9]
        elif calibration_config == 8: self.calib_list = [1,3,7,9,10,12,16,18]
        elif calibration_config == 9: self.calib_list = [i for i in range(1, 10)]
        elif calibration_config == 18: self.calib_list = [i for i in range(1, 19)]
        elif calibration_config == 19: self.calib_list = [i for i in range(0, 19)]
        else: raise ValueError('Invalid calibration configuration')

        self.raw_input = raw_input
        self.gel_spin_max_deg = gel_spin_max_deg if augment else 0.0
        self.center_crop = center_crop
        self.depth_from_npy = depth_from_npy
        self.gel_view_m = gel_view_m

        # 找出所有 unit：<obj>/session_*/sensor_*（找不到就退回平鋪 sensor_*）
        import glob as _glob
        units = sorted(_glob.glob(osp.join(path, '*', 'session_*', 'sensor_*')))
        if not units:
            units = sorted(_glob.glob(osp.join(path, 'sensor_*')))
        units = [u for u in units
                 if osp.isdir(osp.join(u, 'samples')) and osp.isdir(osp.join(u, 'calibration'))]
        # 按物體過濾：unit 路徑 = .../<obj>/session_xxx/sensor_xxxx → 取 <obj>
        if include_objects is not None:
            incl = set(include_objects)
            units = [u for u in units
                     if osp.basename(osp.dirname(osp.dirname(u))) in incl]
        if not units:
            raise RuntimeError(f'在 {path} 找不到任何 unit（include_objects={include_objects}）')
        self.units = units
        self.objects = sorted({osp.basename(osp.dirname(osp.dirname(u))) for u in units})

        # 每個 unit 的 sample 數（取第一個 unit；排除 *_gt）
        samp = [f for f in os.listdir(osp.join(units[0], 'samples'))
                if f.endswith('.png') and '_gt' not in f]
        avail = len(samp)
        self.samples_per_unit = min(num_samples, avail) if num_samples else avail
        self.num_sensors = len(units)        # 對外語意：unit 數
        self.num_samples = self.samples_per_unit

        # 增強 pipeline
        self.tactile_aug = TactileAugment(augment_params) if tactile_augment else None
        extra = {f'c{i}': 'image' for i in range(0, 19)}
        if augment and not tactile_augment:
            self.augment = A.Compose([A.ColorJitter(brightness=(0.6, 1.2), contrast=(0.8, 1.2),
                                                    saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                                      A.Blur()],
                                     additional_targets=extra)
        else:
            self.augment = A.Compose([], additional_targets=extra)

    def __len__(self) -> int:
        return len(self.units) * self.samples_per_unit

    def _get_calib(self, unit):
        """讀取並快取一個 unit 的 calibration（ref + 18 張）。同 unit 的 calib 固定，
        只讀一次，之後所有 sample 重用，大幅減少磁碟 I/O。"""
        cached = self._calib_cache.get(unit)
        if cached is None:
            cal_dir = osp.join(unit, 'calibration')
            ref_path = osp.join(cal_dir, '0000.png')
            if osp.exists(ref_path):
                ref = np.array(Image.open(ref_path))
                calib = [np.array(Image.open(osp.join(cal_dir, '{0:04}.png'.format(i))))
                         for i in range(1, 19)]
            else:
                ref = None
                calib = []
            cached = (ref, calib)
            self._calib_cache[unit] = cached
        return cached

    def getitem_helper(self, unit, sample_idx):
        ref_img, calib_raw = self._get_calib(unit)

        sample = np.array(Image.open(osp.join(unit, 'samples', '{0:04}.png'.format(sample_idx))))

        if self.skip_calibration or ref_img is None:
            sample_f = sample.astype(np.float32)
            if not self.raw_input and ref_img is not None:
                sample_f = sample_f - ref_img.astype(np.float32)
            calib_imgs = []
        elif self.raw_input:
            # Raw mode: no background subtraction, ref included in calibration
            sample_f = sample.astype(np.float32)
            all_imgs = [ref_img.astype(np.float32)] + [c.astype(np.float32) for c in calib_raw]
            calib_imgs = [all_imgs[i] for i in self.calib_list]
        elif self.tactile_aug is not None:
            ref_f = ref_img.astype(np.float32)
            sample_f = sample.astype(np.float32) - ref_f
            calib_imgs = [(calib_raw[i - 1].astype(np.float32) - ref_f)
                           for i in self.calib_list]
        else:
            augments = self.augment(image=sample,
                                    c0=ref_img,
                                    c1=calib_raw[0], c2=calib_raw[1], c3=calib_raw[2],
                                    c4=calib_raw[3], c5=calib_raw[4], c6=calib_raw[5],
                                    c7=calib_raw[6], c8=calib_raw[7], c9=calib_raw[8],
                                    c10=calib_raw[9], c11=calib_raw[10], c12=calib_raw[11],
                                    c13=calib_raw[12], c14=calib_raw[13], c15=calib_raw[14],
                                    c16=calib_raw[15], c17=calib_raw[16], c18=calib_raw[17])
            ref_f = np.array(augments['c0'], dtype=np.float32)
            sample_f = augments['image'].astype(np.float32) - ref_f
            calib_imgs = [(augments[f'c{i}'].astype(np.float32) - ref_f)
                           for i in self.calib_list]

        depth = normal = None
        if self.depth_from_npy:
            try:
                npy_path = osp.join(unit, 'raw_data',
                                    '{0:04}{1}.npy'.format(sample_idx, self.norm_suffix))
                depth = np.load(npy_path).astype(np.float32)
            except Exception:
                pass
            try:
                norm_path = osp.join(unit, 'norms', '{0:04}{1}.png'.format(sample_idx, self.norm_suffix))
                normal = np.array(Image.open(norm_path), dtype=np.float32)
            except Exception:
                if depth is not None:
                    pixel_size = self.gel_view_m * FIXED_CROP / depth.shape[0]
                    normal_raw = depth_to_normal(depth, pixel_size, pixel_size)
                    normal = ((normal_raw + 1.0) * 127.5).clip(0, 255).astype(np.float32)
        else:
            try:
                dmap_path = osp.join(unit, 'dmaps', '{0:04}{1}.png'.format(sample_idx, self.norm_suffix))
                depth = np.array(Image.open(dmap_path), dtype=np.float32)
                norm_path = osp.join(unit, 'norms', '{0:04}{1}.png'.format(sample_idx, self.norm_suffix))
                normal = np.array(Image.open(norm_path), dtype=np.float32)
            except Exception:
                pass

        # gel-spin rotation (before center crop, matching VisTacFusion)
        rot_deg = 0.0
        if self.gel_spin_max_deg > 0:
            rot_deg = np.random.uniform(-self.gel_spin_max_deg, self.gel_spin_max_deg)
            sample_f, calib_imgs, depth, normal = gel_spin_rotate(
                sample_f, calib_imgs, depth, normal, rot_deg)

        # fixed center crop (applied to ALL samples, train+val)
        if self.center_crop:
            sample_f = fixed_center_crop(sample_f)
            calib_imgs = [fixed_center_crop(c) for c in calib_imgs]
            if depth is not None:
                depth = fixed_center_crop(depth)
            if normal is not None:
                normal = fixed_center_crop(normal)

        # photometric augmentation (no geometric since gel_spin handles rotation)
        if self.tactile_aug is not None:
            sample_f, calib_imgs, depth, normal = self.tactile_aug(
                sample_f, calib_imgs, depth, normal)

        # VisTacFusion convention:
        #   depth  = .npy × 1000
        #   normal = PNG / 127.5 - 1.0 → [-1, 1]
        if self.depth_from_npy and depth is not None:
            depth = depth * 1000.0
            if normal is not None:
                normal = normal / 127.5 - 1.0

        sample_t = self.transforms(sample_f)
        calib_t = torch.cat([self.transforms(c) for c in calib_imgs]) if calib_imgs else torch.empty(0)

        if self.depth_from_npy:
            dmap_t = torch.from_numpy(depth).unsqueeze(0) if depth is not None else None
            norm_t = torch.from_numpy(np.ascontiguousarray(normal)).permute(2, 0, 1).float() \
                     if normal is not None else None
        else:
            dmap_t = self.dmap_transforms(depth) if depth is not None else None
            norm_t = self.norm_transforms(normal) if normal is not None else None

        return sample_t, calib_t, dmap_t, norm_t

    def __getitem__(self, index):
        if index >= len(self): raise IndexError(f"Index {index} out of range")
        unit_idx   = index // self.samples_per_unit
        sample_idx = index % self.samples_per_unit
        unit = self.units[unit_idx]

        sample, calib, dmap, norm = self.getitem_helper(unit, sample_idx)

        if self.sendTwo:
            # 第二個 view = 同一張 sample 的另一次隨機增強（augment=False 時兩者相同）
            sample2, calib2, dmap2, norm2 = self.getitem_helper(unit, sample_idx)
            calib  = torch.stack([calib,  calib2],  dim=0)
            sample = torch.stack([sample, sample2], dim=0)
            dmap   = torch.stack([dmap,   dmap2],   dim=0)
            norm   = torch.stack([norm,   norm2],   dim=0)

        # label 取全域唯一 index → SupCon 退化為 SimCLR（只有自己的兩個 view 互為正樣本）
        idx = torch.tensor(index)
        return {'sample': sample, 'calibration': calib, 'dmap': dmap, 'norm': norm, 'idx': idx}


class classification_dataset(Dataset):
    """
    Custom dataset class for classification tasks.
    Handles loading and preprocessing of sensor data with calibration images.
    """
    def __init__(self, 
                 path, 
                 augment=False,
                 transforms=T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)]),
                 calibration_config=18,
                 sensor_list = [],
                 class_list = [],
                 num_samples = None,
                 ) -> None:
        self.path = path
        self.transforms = transforms
        self.sensor_list = sensor_list
        
        # Configure calibration list based on calibration_config parameter
        if calibration_config == 0: self.calib_list = []
        elif calibration_config == 4: self.calib_list = [1,3,7,9]
        elif calibration_config == 8: self.calib_list = [1,3,7,9,10,12,16,18]
        elif calibration_config == 9: self.calib_list = [i for i in range(1, 10)]
        elif calibration_config == 18: self.calib_list = [i for i in range(1, 19)]
        else: raise ValueError('Invalid calibration configuration')
        
        # Clean up directory listings by removing .DS_Store files
        sensors = os.listdir(path)
        if '.DS_Store' in sensors: sensors.remove('.DS_Store')
        
        calibrations = os.listdir(osp.join(path, sensors[0], 'calibration'))
        if '.DS_Store' in calibrations: calibrations.remove('.DS_Store')

        classes = os.listdir(osp.join(path, sensors[0], 'samples'))
        if '.DS_Store' in classes: classes.remove('.DS_Store')
        
        samples =  os.listdir(osp.join(path, sensors[0], 'samples', classes[0]))
        if '.DS_Store' in samples: samples.remove('.DS_Store')
        
        # Set dataset dimensions
        self.num_sensors = len(sensor_list)
        self.num_calibrations = len(calibrations)
        self.num_classes = len(classes)
        
        if num_samples is not None: self.num_samples = num_samples
        else: self.num_samples = len(samples)
        
        if class_list:
            self.num_classes = len(class_list)
        self.class_list = class_list
        
        # Initialize augmentation pipeline
        self.augment = A.Compose([], additional_targets={'c0':'image', 
                                                         'c1':'image', 'c2':'image', 'c3':'image', 
                                                         'c4':'image', 'c5':'image', 'c6':'image', 
                                                         'c7':'image', 'c8':'image', 'c9':'image', 
                                                         'c10':'image', 'c11':'image', 'c12':'image', 
                                                         'c13':'image', 'c14':'image', 'c15':'image', 
                                                         'c16':'image', 'c17':'image', 'c18':'image'})
        if augment == True:
            # Configure data augmentation transforms
            self.augment = A.Compose([A.ColorJitter(brightness=(0.6, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                                      A.Blur()],
                                      additional_targets={'c0':'image', 
                                                          'c1':'image', 'c2':'image', 'c3':'image', 
                                                          'c4':'image', 'c5':'image', 'c6':'image', 
                                                          'c7':'image', 'c8':'image', 'c9':'image', 
                                                          'c10':'image', 'c11':'image', 'c12':'image', 
                                                          'c13':'image', 'c14':'image', 'c15':'image', 
                                                          'c16':'image', 'c17':'image', 'c18':'image'})

    def __len__(self) -> int:
        """Return total number of samples in the dataset"""
        return self.num_samples * self.num_classes * self.num_sensors
    
    def getitem_helper(self, sensor_idx, class_idx, sample_idx):    
        """
        Helper function to load and preprocess a single sample
        Returns processed sample, calibration data, and label
        """
        # Load background reference image for subtraction
        ref_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '0000.png')
        ref_img = np.array(Image.open(ref_path))
        
        # Load all calibration images
        calib = []
        for i in range(1,19):
            calib_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '{0:04}.png'.format(i))
            calib_img = np.array(Image.open(calib_path))
            calib.append(calib_img)
        
        # Load the actual sample image
        sample_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'samples', 'class_{0:04}'.format(class_idx), '{0:04}.png'.format(sample_idx))
        sample = np.array(Image.open(sample_path)) 
        
        # Apply augmentations to all images (sample and calibration)
        augments = self.augment(image=sample,
                                c0=ref_img,
                                c1=calib[0], c2=calib[1], c3=calib[2],
                                c4=calib[3], c5=calib[4], c6=calib[5], 
                                c7=calib[6], c8=calib[7], c9=calib[8],
                                c10=calib[9], c11=calib[10], c12=calib[11],
                                c13=calib[12], c14=calib[13], c15=calib[14],
                                c16=calib[15], c17=calib[16], c18=calib[17])
        
        # Process reference image
        ref_img = np.array(augments['c0'], dtype=np.float32)
        
        # Process calibration images: subtract background and apply transforms
        calib = torch.tensor([])
        for i in self.calib_list:
            calib_img = np.array(augments[f'c{i}'], dtype=np.float32)
            calib_img = augments[f'c{i}'] - ref_img
            calib = torch.cat([calib, self.transforms(calib_img)])
            
        # Process sample image: subtract background and apply transforms
        sample =  np.array(augments['image'], dtype=np.float32)
        sample = augments['image'] - ref_img
        sample = self.transforms(sample) 
        
        # Create one-hot encoded label
        if self.class_list:
            label = torch.nn.functional.one_hot(torch.tensor(self.class_list.index(class_idx)), num_classes=self.num_classes)
        else: label = torch.nn.functional.one_hot(torch.tensor(class_idx), num_classes=self.num_classes)
        label = label.float()
            
        return sample, calib, label
    
    def __getitem__(self, index):
        """
        Main method to get a sample from the dataset
        Returns a dictionary containing the sample, calibration data, and label
        """
        if index >= len(self): raise IndexError(f"Index {index} out of range")
        
        # Calculate indices for sensor, class, and sample
        sensor_idx = self.sensor_list[index // (self.num_samples * self.num_classes)]
        class_idx = (index % (self.num_samples * self.num_classes)) // self.num_samples
        if self.class_list:
            class_idx = self.class_list[class_idx]
        sample_idx = index % self.num_samples
        
        # Get the processed data
        sample, calib, label = self.getitem_helper(sensor_idx, class_idx, sample_idx)
        
        return {'sample': sample, 'calibration': calib, 'label': label}

class pose_dataset(Dataset):
    """
    Custom dataset class for pose estimation that handles sensor data with background subtraction.
    Supports data augmentation and multiple sensor inputs.
    """
    def __init__(self, 
                 path, 
                 augment=False,
                 transforms=T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)]),
                 sensor_list = [],
                 calibration_config = 18,
                 random_final = False,
                 ) -> None:
        self.path = path
        self.transforms = transforms
        self.sensor_list = sensor_list
        self.random_final = random_final
        
        if calibration_config == 0: self.calib_list = []
        elif calibration_config == 4: self.calib_list = [1,3,7,9]
        elif calibration_config == 8: self.calib_list = [1,3,7,9,10,12,16,18]
        elif calibration_config == 9: self.calib_list = [i for i in range(1, 10)]
        elif calibration_config == 18: self.calib_list = [i for i in range(1, 19)]
        else: raise ValueError('Invalid calibration configuration')
        
        # Get list of sensors, calibrations, and samples from the data directory
        sensors = os.listdir(path)
        if '.DS_Store' in sensors: sensors.remove('.DS_Store')
        
        calibrations = os.listdir(osp.join(path, sensors[0], 'calibration'))
        if '.DS_Store' in calibrations: calibrations.remove('.DS_Store')
        
        classes = os.listdir(osp.join(path, sensors[0], 'samples'))
        if '.DS_Store' in classes: classes.remove('.DS_Store')
        
        samples =  os.listdir(osp.join(path, sensors[0], 'samples', classes[0]))
        if '.DS_Store' in samples: samples.remove('.DS_Store')
        
        # Store counts for dataset size calculation
        self.num_sensors = len(sensor_list)
        self.num_calibrations = len(calibrations)
        self.num_classes = len(classes)
        self.num_samples = len(samples)
        
        self.augment = A.Compose([], additional_targets={'c0':'image', 
                                                         'c1':'image', 'c2':'image', 'c3':'image', 
                                                         'c4':'image', 'c5':'image', 'c6':'image', 
                                                         'c7':'image', 'c8':'image', 'c9':'image', 
                                                         'c10':'image', 'c11':'image', 'c12':'image', 
                                                         'c13':'image', 'c14':'image', 'c15':'image', 
                                                         'c16':'image', 'c17':'image', 'c18':'image'})
        if augment == True:
            # we need batched augmentations for calibration
            self.augment = A.Compose([A.ColorJitter(brightness=(0.6, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                                      A.Blur()],
                                      additional_targets={'c0':'image', 
                                                          'c1':'image', 'c2':'image', 'c3':'image', 
                                                          'c4':'image', 'c5':'image', 'c6':'image', 
                                                          'c7':'image', 'c8':'image', 'c9':'image', 
                                                          'c10':'image', 'c11':'image', 'c12':'image', 
                                                          'c13':'image', 'c14':'image', 'c15':'image', 
                                                          'c16':'image', 'c17':'image', 'c18':'image'})

    def __len__(self) -> int:
        return self.num_samples * self.num_classes * self.num_sensors
    
    def getitem_helper(self, sensor_idx, class_idx, sample_idx):    
        """
        Helper method to load and preprocess a single sample from the dataset.
        Handles background subtraction and data augmentation.
        """
        # Load background reference image for subtraction
        ref_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '0000.png')
        ref_img = np.array(Image.open(ref_path))
        
        # Load calibration images
        calib = []
        for i in range(1,19):
            calib_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '{0:04}.png'.format(i))
            calib_img = np.array(Image.open(calib_path))
            calib.append(calib_img)
            
        # Load sample image
        sample_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'samples', 'obj_{0:04}'.format(class_idx), '{0:04}.png'.format(sample_idx))
        sample = np.array(Image.open(sample_path)) 
        
        # Apply augmentations to all images
        augments = self.augment(image=sample,
                                c0=ref_img,
                                c1=calib[0], c2=calib[1], c3=calib[2],
                                c4=calib[3], c5=calib[4], c6=calib[5], 
                                c7=calib[6], c8=calib[7], c9=calib[8],
                                c10=calib[9], c11=calib[10], c12=calib[11],
                                c13=calib[12], c14=calib[13], c15=calib[14],
                                c16=calib[15], c17=calib[16], c18=calib[17])
        
        # Process reference image
        ref_img = np.array(augments['c0'], dtype=np.float32)
        
        # Process calibration images with background subtraction and normalization
        calib = torch.tensor([])
        for i in self.calib_list:
            calib_img = np.array(augments[f'c{i}'], dtype=np.float32)
            calib_img = augments[f'c{i}'] - ref_img
            calib = torch.cat([calib, self.transforms(calib_img)])
            
        # Process sample image with background subtraction and normalization
        sample =  np.array(augments['image'], dtype=np.float32)
        sample = augments['image'] - ref_img
        sample = self.transforms(sample) 
        
        # Load location data
        location_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'locations', 'obj_{0:04}'.format(class_idx), '{0:04}.npy'.format(sample_idx))
        location = np.load(location_path)[:3]
        location = torch.tensor(location, dtype=torch.float32)
        
        return sample, calib, location
    
    def __getitem__(self, index):
        """
        Returns a data sample containing initial and final states, calibration data, and location change.
        """
        if index >= len(self): raise IndexError(f"Index {index} out of range")
        
        sensor_idx = self.sensor_list[index // (self.num_samples * self.num_classes)]
        class_idx = (index % (self.num_samples * self.num_classes)) // self.num_samples
        sample_idx = index % self.num_samples
        
        # Get initial state data
        sample_init, calib, location_init = self.getitem_helper(sensor_idx, class_idx, sample_idx)
        
        # Get final state data (either next sequential sample or random sample)
        if self.random_final: next_index = random.randint(0,self.num_samples-1)
        else: next_index = (index+1)%self.num_samples
        
        sample_final, _, location_final = self.getitem_helper(sensor_idx, class_idx, next_index)
        
        # Calculate location change between initial and final states
        location = location_final - location_init
        
        return {'sample_init': sample_init, 'sample_final': sample_final, 'calibration': calib, 'label': location, 'sensor': sensor_idx, 'idx':sample_idx, 'class':class_idx}
  
# Script to calculate dataset statistics (mean and std) for normalization
if __name__ == '__main__':
    # Initialize dataset without normalization for statistics calculation
    ds = sim_dataset(
        transforms=T.Compose([T.ToTensor()]), 
        dmap_transforms=T.Compose([T.ToTensor()]),
        norm_transforms=T.Compose([T.ToTensor()]),
    )
    
    # Initialize accumulators for statistics calculation
    rgb_sum = torch.tensor([0.0, 0.0, 0.0])
    rgb_sum_sq = torch.tensor([0.0, 0.0, 0.0])

    dmap_sum = torch.tensor([0.0])
    dmap_sum_sq = torch.tensor([0.0])

    norm_sum = torch.tensor([0.0, 0.0, 0.0])
    norm_sum_sq = torch.tensor([0.0, 0.0, 0.0])

    # Calculate running sums for mean and variance
    pbar = tqdm(ds)    
    for i, batch in enumerate(pbar):
        sample = batch['sample']
        dmap = batch['dmap']
        norm = batch['norm']
        
        rgb_sum += sample.sum(axis=[1,2])
        rgb_sum_sq += (sample**2).sum(axis=[1,2])

        dmap_sum += dmap.sum()
        dmap_sum_sq += (dmap**2).sum()

        norm_sum += norm.sum(axis=[1,2])
        norm_sum_sq += (norm**2).sum(axis=[1,2])

    # Calculate final statistics
    count = len(ds) * 224 * 224 

    rgb_mean = rgb_sum / count
    rgb_var = (rgb_sum_sq / count) - (rgb_mean**2)
    rgb_std = torch.sqrt(rgb_var)

    dmap_mean = dmap_sum / count
    dmap_var = (dmap_sum_sq / count) - (dmap_mean**2)
    dmap_std = torch.sqrt(dmap_var)

    norm_mean = norm_sum / count
    norm_var = (norm_sum_sq / count) - (norm_mean**2)
    norm_std = torch.sqrt(norm_var)
    
    # Print calculated statistics
    print("rgb mean: " + str(rgb_mean))
    print("rgb std:  " + str(rgb_std))
    print('')
    print("dmap mean: " + str(dmap_mean))
    print("dmap std:  " + str(dmap_std))
    print('')
    print("norm mean: " + str(norm_mean))
    print("norm std:  " + str(norm_std))
    