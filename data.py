import torch
import pandas as pd
import glob
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils import rle_decode, get_mask
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, img_size, num_classes=4, transforms=None):
        self.folder_path = folder_path
        self.image_size = img_size
        self.num_classes = num_classes
        self.df = pd.read_csv("imaterialist-fashion-2019-FGVC6/train.csv")
        self.transforms = transforms
        
        self.df['CategoryId'] = self.df['ClassId'].str.split('_').str[0].astype(int)

        self.df['CategoryId'] = self.df['CategoryId'] % num_classes

        self.df = self.df.groupby("CategoryId").head(500)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]
        img = Image.open(os.path.join(self.folder_path, image_id)).convert("RGB")
        mask = get_mask(self.df, image_id, self.num_classes)

        img = np.array(img.resize((self.image_size, self.image_size)))
        mask = np.array(mask.resize((self.image_size, self.image_size)))
        mask[mask >= self.num_classes] = self.num_classes

        if self.transforms:
            transform = self.transforms(image=img, masked=mask)
            img, mask = transform["image"], transform["mask"]
            img = torch.from_numpy(img).permute(2, 0, 1) / 255
            mask = torch.from_numpy(mask)
        return img / 255, mask

    
    #     
    #     self.img_size = img_size
    #     self.df = pd.read_csv("imaterialist-fashion-2019-FGVC6/train.csv")
    #     self.transforms = transforms
    #     self.df['CategoryId'] = self.df["ClassId"].apply(lambda x: str(x).split("_")[0])
    #     self.num_classes = self.df['CategoryId'].nunique()
        self.df = self.df.groupby('ImageId')['EncodedPixels', 'CategoryId', 'Height', 'Width'].agg(lambda x: list(x)).reset_index()
        self.df["Height"] = self.df["Height"].apply(lambda x: np.mean(x)).astype(int)
        self.df["Width"] = self.df["Width"].apply(lambda x: np.mean(x)).astype(int)
    

    # def __getitem__(self, idx):
    #     imagePath = os.path.join(self.folder_path, self.df["ImageId"].iloc[idx])
    #     img = cv2.imread(imagePath)
    #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, self.img_size)

    #     info = self.df.iloc[idx]

    #     mask = np.zeros((self.img_size[1], self.img_size[0], self.num_classes))
    #     for annotation, label in zip(info["EncodedPixels"], info['CategoryId']):
    #         height, width = info["Height"], info["Width"]
    #         cur_mask = rle_decode(annotation, (height, width))
    #         mask[:, :, int(label)] += cv2.resize(cur_mask, self.img_size)
            
    #     mask = (mask > 0.5).astype(np.float32)
    #     if self.transforms is not None:
    #         transform = self.transforms(image=img, mask=mask)

    #         img, mask = transform["image"], transform["mask"]
    #         img = torch.from_numpy(img).permute(2, 0, 1)
    #         mask = torch.from_numpy(mask).permute(2, 0, 1)
        
    #     return img, mask

    # def __len__(self):
    #     return len(self.df)

