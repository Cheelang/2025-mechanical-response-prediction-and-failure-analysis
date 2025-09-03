import numpy as np
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        size = [self.size, self.size]
        image = F.resize(image, size)
        return image

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
        return image

class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return F.normalize(image, mean=self.mean, std=self.std)

if __name__ == '__main__':
    from PIL import Image
    img = Image.open('123.tif').convert('RGB')
    img = Resize(256)(img)
    img_tensor = ToTensor()(img)
    img_tensor = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img_tensor)
    print(img_tensor.shape)