import monai
import osail_utils
import copy
import torchvision
import numpy as np

##########################
## Transforms
##########################

def build_train_transforms(**kwargs):
    
    img_height = kwargs.get("img_height", 1024)
    img_width = kwargs.get("img_width", 1024)

    train_trans = monai.transforms.Compose(
        [
            # osail_utils.io.LoadImageD(
            #     keys=["image", "label"], 
            #     pad=True, 
            #     target_shape=None, 
            #     transpose=False, 
            #     ensure_grayscale=False
            # ),
            monai.transforms.LoadImageD(keys=["image", "label"], image_only=True, prune_meta_pattern=".*|.*", prune_meta_sep=" "),
            monai.transforms.TransposeD(keys=["image", "label"], indices=[2,1,0]),
            Pad2Square(keys=["image", "label"]),
            monai.transforms.ResizeD(keys=["image", "label"], spatial_size=(img_height, img_width), mode="nearest"),
            monai.transforms.NormalizeIntensityD(keys=["image"]),
            monai.transforms.ScaleIntensityD(keys=["image"]),
            monai.transforms.RandFlipD(keys=["image", "label"], prob=0.5),
            monai.transforms.RandGaussianNoiseD(keys=["image"], prob=0.5),
            monai.transforms.RandRotateD(keys=["image", "label"], prob=0.5, range_x=0.6, mode="nearest"),
            monai.transforms.RandAdjustContrastD(keys=["image"], gamma=(0.2,4))
        ]
    )
    
    return train_trans

def build_val_transforms(**kwargs):
    
    img_height = kwargs.get("img_height", 1024)
    img_width = kwargs.get("img_width", 1024)
    
    
    val_trans = monai.transforms.Compose(
        [
            # osail_utils.io.LoadImageD(
            #     keys=["image", "label"], 
            #     pad=True, 
            #     target_shape=(img_height, img_width), 
            #     transpose=True, 
            # ),
            # #monai.transforms.LoadImageD(keys=["image", "label"], image_only=True, prune_meta_pattern=".*|.*", prune_meta_sep=" "),
            # #monai.transforms.TransposeD(keys=["image", "label"], indices=[2,1,0]),
            # #monai.transforms.ResizeD(keys=["image", "label"], spatial_size=(img_height, img_width), mode="nearest"),
            # #monai.transforms.ScaleIntensityD(keys=["image"]),
            monai.transforms.LoadImageD(keys=["image", "label"], image_only=True, prune_meta_pattern=".*|.*", prune_meta_sep=" "),
            monai.transforms.TransposeD(keys=["image", "label"], indices=[2,1,0]),
            Pad2Square(keys=["image", "label"]),
            monai.transforms.ResizeD(keys=["image", "label"], spatial_size=(img_height, img_width), mode="nearest"),
            monai.transforms.NormalizeIntensityD(keys=["image"]),
            monai.transforms.ScaleIntensityD(keys=["image"]),
        ]
    )
    
    return val_trans

class Pad2Square(monai.transforms.Transform):
    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        data_copy = copy.deepcopy(data)
        for key in self.keys:
            if key in data:
                img = data[key]
                s = list(img.size())
                max_wh = np.max([s[-1], s[-2]])
                hp = int((max_wh - s[-1]) / 2)
                vp = int((max_wh - s[-2]) / 2)
                padding = (hp, vp, hp, vp)
                img = torchvision.transforms.functional.pad(img, padding, 0, "constant")
                data_copy[key] = img
        return data_copy
