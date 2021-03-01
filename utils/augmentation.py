import numpy as np
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma,
    ChannelShuffle, RGBShift, RandomRotate90, IAAAdditiveGaussianNoise, GaussNoise
)

def augmentation(img, mask, color_aug_prob):
    def aug(p1, color_aug_prob):
        return Compose([
            RandomRotate90(p=0.8),
            Flip(p=0.6),
            OneOf([
                IAAAdditiveGaussianNoise(p=1), 
                HueSaturationValue(p=1), 
                CLAHE(p=1),
                OpticalDistortion(p=1), 
                RandomContrast(p=1),
                RandomBrightness(p=1),
                RandomGamma(p=1),
                IAAEmboss(p=1),
                MotionBlur(p=1),
            ], p=color_aug_prob)
        ], p=p1)
    #image = img.transpose(2,0,1)
    mask = mask.astype('uint8')
    aug_func = aug(0.9, color_aug_prob)
    
    data = {"image": img, "mask": mask}#img.astype(np.float32)/255
    augmented = aug_func(**data)
    image, mask = augmented["image"], augmented["mask"]

    return image, mask
