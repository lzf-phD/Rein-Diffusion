import glob
import os
import cv2
import torchvision
import torch
import numpy as np
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

class ReinDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """

    def __init__(self, split, im_path, im_h=1024,im_w=512, im_channels=3, im_ext='png',
                 use_latents=False, latent_path=None, return_hint=False):#, condition_config=None):
        self.split = split
        self.im_h = im_h
        self.im_w = im_w
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        self.return_hints = return_hint
        self.imageA,self.imageB,self.text = self.load_images(im_path)

        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.imageA):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        img_A = glob.glob(os.path.join(im_path, f"mask", "*.png"))
        img_B = glob.glob(os.path.join(im_path, f"{self.split}_B", "*.png"))
        text = glob.glob(os.path.join(im_path, f"cond", "*.txt"))
        img_A = sorted(img_A)
        img_B = sorted(img_B)
        text = sorted(text)
        return img_A,img_B,text

    def __len__(self):
        return len(self.imageA)

    def __getitem__(self, index):

        if self.use_latents:
            fname = os.path.basename(self.imageB[index])
            txt_path = self.text[index]
            txt_embedding = load_txt_as_tensor(txt_path)
            matched_keys = [k for k in self.latent_maps if fname in k]

            if matched_keys:
                latent = self.latent_maps[matched_keys[0]]
            if self.return_hints:
                canny_image = Image.open(self.imageA[index])
                canny_image_tensor = torchvision.transforms.ToTensor()(canny_image)
                canny_image_tensor = canny_image_tensor.repeat(3, 1, 1)
                return latent,txt_embedding,canny_image_tensor
            else:
                return latent,txt_embedding

        else:
            image_B = Image.open(self.imageB[index])
            transform = torchvision.transforms.ToTensor()
            tensor_B = transform(image_B)
            image_B.close()
            tensor_B = (2 * tensor_B) - 1

            if self.return_hints:
                canny_image = Image.open(self.imageA[index])
                canny_image_tensor = torchvision.transforms.ToTensor()(canny_image)
                canny_image_tensor = canny_image_tensor.repeat(3, 1, 1)
                return tensor_B, canny_image_tensor
            else:
                return tensor_B

def load_txt_as_tensor(txt_path):
    values = np.loadtxt(txt_path, dtype=float, encoding="utf-8")
    assert len(values) == 2
    val1 = normalize_parameter(values[0], 0, 30000) #Modification based on the actual situation.
    val2 = normalize_parameter(values[1], 0, 30000) #Modification based on the actual situation.
    vec1 = np.full((768,), val1, dtype=np.float32)
    vec2 = np.full((768,), val2, dtype=np.float32)

    tensor = torch.tensor(np.stack([vec1, vec2], axis=0))
    return tensor

def normalize_parameter(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

