import numpy as np
import pandas as pd
from PIL import Image


def rle_decode(mask_rle, shape):
    '''
    https://www.kaggle.com/code/vladimirsydor/train-unet-mobilenet
    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
    shape: (height,width) of array to return
    Returns numpy array according to the shape, 1 - mask, 0 - background
    '''
    shape = (shape[1], shape[0])
    s = mask_rle.split()
    # gets starts & lengths 1d arrays
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    # gets ends 1d array
    ends = starts + lengths
    # creates blank mask image 1d array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # sets mark pixles
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # reshape as a 2d mask image
    return img.reshape(shape).T  # Needed to align to RLE direction


def get_mask(df, img_id, num_classes):
        # https://www.kaggle.com/code/josutk/show-segmentation
        a = df[df.ImageId == img_id]
        a = a.groupby('CategoryId', as_index=False).agg({'EncodedPixels': ' '.join, 'Height': 'first', 'Width': 'first'})
        H = int(a.iloc[0, 2])
        W = int(a.iloc[0, 3])
        mask = np.full(H * W, dtype='int', fill_value=num_classes)
        for line in a[['EncodedPixels', 'CategoryId']].iterrows():
            encoded = line[1][0]
            pixel_loc = list(map(int, encoded.split(' ')[0::2]))
            iter_num = list(map(int, encoded.split(' ')[1::2]))
            for p, i in zip(pixel_loc, iter_num):
                mask[p: (p + i)] = line[1][1]
        mask = mask.reshape(W, H).T
        mask = Image.fromarray(np.uint8(mask)).convert('L')
        return mask