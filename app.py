import torch
from torch.utils.data import DataLoader
from utils import MyImageDataset


if __name__ == "__main__":

    test_dset = MyImageDataset(img_dir="dataset/test/images/", lbl_dir="dataset/test/labels/") # create dataset
    test_dloader = DataLoader(dataset=test_dset, batch_size=2, shuffle=True) # create dataloader
    test_feature, test_label = next(iter(test_dloader)) # check if it works

