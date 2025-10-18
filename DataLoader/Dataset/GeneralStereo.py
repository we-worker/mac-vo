import cv2
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from torch.utils.data import Dataset

from ..Interface import StereoFrame, StereoData
from ..SequenceBase import SequenceBase


class GeneralStereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "GeneralStereo"
    
    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        
        # metadata
        self.seqRoot = Path(cfg.root)
        self.baseline = cfg.bl
        self.T_BS = pp.identity_SE3(1, dtype=torch.float64)
        
        # 获取目标尺寸，默认为原始尺寸
        target_size = getattr(cfg, 'target_size', None)
        
        self.ImageL    = MonocularDataset(Path(self.seqRoot, "left"), cfg.format, target_size)
        self.ImageR    = MonocularDataset(Path(self.seqRoot, "right"), cfg.format, target_size)
        assert len(self.ImageL) == len(self.ImageR)
        
        if hasattr(cfg.camera, "fx"):
            # 计算缩放比例
            if target_size is not None:
                # 假设原始尺寸为1920x1200
                original_width, original_height = 1920, 1200
                scale_x = target_size[0] / original_width
                scale_y = target_size[1] / original_height
                
                # 调整相机内参
                scaled_fx = cfg.camera.fx * scale_x
                scaled_fy = cfg.camera.fy * scale_y
                scaled_cx = cfg.camera.cx * scale_x
                scaled_cy = cfg.camera.cy * scale_y
            else:
                scaled_fx = cfg.camera.fx
                scaled_fy = cfg.camera.fy
                scaled_cx = cfg.camera.cx
                scaled_cy = cfg.camera.cy
            
            self.K = torch.tensor([[
                [scaled_fx, 0., scaled_cx], 
                [0., scaled_fy, scaled_cy],
                [0.        , 0., 1.       ]
            ]], dtype=torch.float).repeat(len(self.ImageL), 1, 1)
        else:
            K_original = torch.tensor(np.load(Path(self.seqRoot, "intrinsic.npy"))).float()
            if target_size is not None:
                # 从第一张图像获取原始尺寸
                sample_image = cv2.imread(str(self.ImageL.file_names[0]))
                original_height, original_width = sample_image.shape[:2]
                scale_x = target_size[0] / original_width
                scale_y = target_size[1] / original_height
                
                # 调整内参矩阵
                scale_matrix = torch.tensor([
                    [scale_x, 0., 0.],
                    [0., scale_y, 0.],  
                    [0., 0., 1.]
                ], dtype=torch.float)
                self.K = torch.matmul(scale_matrix.unsqueeze(0), K_original)
            else:
                self.K = K_original

        self.length = len(self.ImageL)
        super().__init__(self.length)

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)
        imageL = self.ImageL[index]
        imageR = self.ImageR[index]
            
        return StereoFrame(
            idx    = [local_index],
            time_ns= [local_index * 1000],         # FIXME: a fake timestamp.
            stereo = StereoData(
                T_BS     = self.T_BS,
                K        = self.K[index:index+1],
                baseline = torch.tensor([self.baseline]),
                width    = imageL.size(-1),
                height   = imageL.size(-2),
                time_ns  = [local_index * 1000],   # FIXME: a fake timestamp.
                imageL   = imageL,
                imageR   = imageR
            )
        )

    @classmethod
    def is_valid_config(cls, config) -> None:
        cls._enforce_config_spec(config, {
            "root"  : lambda s: isinstance(s, str),
            "bl"    : lambda v: isinstance(v, float),
            "format": lambda s: isinstance(s, str),
            "target_size": lambda v: v is None or (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) for x in v)),
            "camera": lambda v: isinstance(v, dict) and (len(v) == 0 or cls._enforce_config_spec(v, {
                "fx": lambda v: isinstance(v, float),
                "fy": lambda v: isinstance(v, float),
                "cx": lambda v: isinstance(v, float),
                "cy": lambda v: isinstance(v, float)
            }, allow_excessive_cfg=True)) or True
        })

class MonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    Return the image in shape (1, 3, H, W) with dtype=float32 
    and normalized (image in [0, 1])
    """
    def __init__(self, directory: Path, format: Literal["png", "jpg"], target_size: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.directory = directory
        self.target_size = target_size  # (width, height)
        assert self.directory.exists(), f"Monocular image directory {self.directory} does not exist"
        
        self.file_names = list(sorted(directory.glob(f"*.{format}")))
            
        self.length = len(self.file_names)
        assert self.length > 0, f"No file with '.{format}' suffix is found under {self.directory}"

    @staticmethod
    def apply_white_balance(image: np.ndarray) -> np.ndarray:
        """
        Apply white balance correction to RGB image
        Parameters based on custom calibration:
        R channel: (R - 51 + 1491/23) / (71/92)
        B channel: (B - 82 + 6399/98) / (81/98)
        G channel: no change
        
        Args:
            image: RGB image in range [0, 255] with shape (H, W, 3)
        
        Returns:
            White balanced image in range [0, 255] with shape (H, W, 3)
        """
        image = image.astype(np.float32)
        
        # Apply white balance to R channel (channel 0)
        image[:, :, 0] = (image[:, :, 0] - 51 + (1491/23)) / (71/92)
        
        # G channel (channel 1) remains unchanged
        
        # Apply white balance to B channel (channel 2)
        image[:, :, 2] = (image[:, :, 2] - 82 + (6399/98)) / (81/98)
        
        # Clip to valid range [0, 255]
        image = np.clip(image, 0, 255)
        
        return image.astype(np.uint8)

    @staticmethod
    def load_png_format(path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(f"Failed to read image from {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Output image tensor in shape of (1, C, H, W)
        image = self.load_png_format(str(self.file_names[index]))
        
        # Apply white balance correction
        image = self.apply_white_balance(image)
        
        # 如果指定了目标尺寸，则进行缩放
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image /= 255.
        return image
