# dataset with augmentation

import enum
import albumentations as albu
import cv2
import os
import re
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image


class AugDataset(object):
    def __init__(self, img_root, augmentations=[], transform=None):
        """
        init dataset with a specified pix4d project folder.

        Parameters
        ------------------
        img_root: このディレクトリ下にクラス別のディレクトリ `${img_root}/*_${class_id}/` がある
        augmentations: [albu.Transoform]
        """
        self.augmentations = [None, *augmentations]
        self.aug_cnt = len(self.augmentations)
        self.transform = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.img_root = img_root
        class_dirs = list(sorted(
            [e for e in os.scandir(img_root) if e.is_dir()],
            key=lambda e: e.path
        ))
        print(f"{len(class_dirs)} class found in {img_root}")
        self.class_list = [(
            e.path,
            list(sorted(os.listdir(e.path)))
        ) for e in class_dirs]
        self.class_cnt = []
        for class_idx, c in enumerate(self.class_list):
            class_path, files = c
            print(f"class idx:{class_idx} dir:{class_path} img cnt:{len(files)}")
            self.class_cnt.append(len(files))
        print(f"augmentation x {len(self.augmentations)}")
        self.img_cnt = sum(self.class_cnt)
        print(f"img size: {self.img_cnt}")
        print(f"data size: {self.img_cnt * self.aug_cnt}")

    def __getitem__(self, idx):
        aug_idx = idx % self.aug_cnt
        img_idx = int(idx / self.aug_cnt)
        class_idx = 0
        while self.class_cnt[class_idx] <= img_idx:
            img_idx -= self.class_cnt[class_idx]
            class_idx += 1
        class_dir, files = self.class_list[class_idx]
        # load images and masks
        img_path = os.path.join(
            class_dir, files[img_idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = class_idx

        # np array
        transform = self.augmentations[aug_idx]
        if transform is not None:
            img = transform(image=img)['image']

        # PIL Image
        img = Image.fromarray(img)
        transform = self.transform
        if transform is not None:
            img = transform(img)

        return img, label

    def __len__(self):
        return self.img_cnt * self.aug_cnt

    def num_class(self):
        return len(self.class_list)

# https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/5
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

# each of augmentation will be applied
augs = [
    albu.HorizontalFlip(),
    albu.Blur(blur_limit=5, p=1),
    albu.RandomBrightnessContrast(),
    albu.RandomGamma(gamma_limit=(80, 120)),
    albu.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=1),
    albu.GridDistortion(num_steps=5, distort_limit=0.3,
                        interpolation=1, border_mode=4, p=1),
    albu.RandomResizedCrop(256, 256, scale=(0.5, 0.9),
                           ratio=(1.0, 1.0), interpolation=1)
]

# this transform is applied for all the images after augmentation
transform = transforms.Compose(
    [
        SquarePad(),  # padding for square
        transforms.Resize(224),  # for ResNet50
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # normalization for each channels
    ]
)

dataset = AugDataset('data', transform=transform)
