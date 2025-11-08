from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torch.utils.data.dataset import Dataset
from PIL import Image

from os import listdir
from os.path import join


class DatasetFromFolder(Dataset):
    # initialisation of class, image names are stored and transformations are calculated from super
    def __init__(self, data_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_names = [join(data_dir, x) for x in listdir(data_dir)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    # for every item, image is converted to YCbCr notation; since Cb and Cr are constants
    # only Y value is passed to be transformed
    def __getitem__(self, i):
        image, _, _, = Image.open(self.image_names[i]).convert('YCbCr').split()
        target = image.copy()
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.image_names)


def round_crop_size(crop_size, upscale):
    return crop_size - (crop_size % upscale)


def input_transform(crop_size, upscale):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


# the variables are passed from train.py
def set_maker(directory, crop_amount, upscale):
    crop_size = round_crop_size(crop_amount, upscale)
    return DatasetFromFolder(directory, input_transform=input_transform(crop_size, upscale),
                             target_transform=target_transform(crop_size))