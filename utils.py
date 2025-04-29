import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

class MyImageDataset(Dataset):
    
    def __init__(self, img_dir:str, lbl_dir:str, img_size:tuple=(448, 448), S:int=7, C:int=2):

        super().__init__()
        
        self.S = S # grid parameter (cells number = SxS)
        self.C = C # number of classes
        self.img_size = img_size # image size used by yolo (in v1 (448x448))

        # define image transforms - values from imageNet
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(), # float tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalization
            transforms.Resize(img_size), # scale to (448, 448)
        ])

        # get label and image files (imgs only if they are annotated)
        self.lbl_files, self.img_files = self._get_label_and_image_files(img_dir=img_dir, lbl_dir=lbl_dir)
    
    def __len__(self):

        return len(self.lbl_files)

    def __getitem__(self, idx):

        # get label and image file paths, corresponding to idx
        img_path = self.img_files[idx]
        lbl_path = self.lbl_files[idx]

        # get image and label 
        image = Image.open(img_path)

        label_dims = (self.S, self.S, 5 + self.C) # calculate dims of target tensor 
        label = torch.zeros(size=label_dims, dtype=float) # define target tensor (full of zeros)

        # read annots from file, and convert it to right format for yolo v1 target
        annotations = self._read_annotations_from_txt_file(lbl_path)

        for annotation in annotations: # loop over annotations
            
            # get row and col of a responsible grid cell
            col = int(annotation[-2]) 
            row = int(annotation[-1])

            # insert annots (c, xc, yc, w, h, ohe_vector)
            label[row, col] = torch.Tensor(annotation[:-2])

        # transform image
        image = self.img_transforms(image)

        return image, label
    
    def _get_label_and_image_files(self, img_dir:str, lbl_dir:str) -> tuple[list[Path]]:
        
        img_suffixes: list[str] = [".jpg", ".jpeg", ".gif", ".png"] # accepted image formats (by torchvision.io.image_decode)
        
        # get absolute paths to images and labels (in yolo .txt format)
        img_path = Path.cwd()/img_dir 
        lbl_path = Path.cwd()/lbl_dir

        lbl_files = [lbl for lbl in lbl_path.iterdir() if lbl.is_file()] # label files
        lbl_stems = [nm.stem for nm in lbl_files] # filenames without path and extension
        img_files = [img for img in img_path.iterdir() if (img.is_file() and img.suffix in img_suffixes and img.stem in lbl_stems)] # image files (only ones with corresponding label .txt file)
        return lbl_files, img_files

    def _read_annotations_from_txt_file(self, filename):
        annots: list = []
        with open(file=filename, mode='r') as fn:
            for line in fn:
                present_annot = line.strip().split(' ') # get single annotation line into a list of strings 
                annots.append([float(element) for element in present_annot]) # convert into float and append to annots list (of lists)
        
        # convert (xc, yc) into absolute (pixels), global values
        for annot in annots: 
            annot[1:3] = self._global_relative_to_absolute(annot[1:3], self.img_size) # convert (xc, yc) into global, absolute values (pixels)
            annot[1],annot[2],ind_x,ind_y = self._global_absolute_to_local_relative(annot[1:3], self.img_size, self.S) # convert xc, yc to local, relative values
            real_cat = int(annot[0]) # get real object's category
            ohe_vector = self._prepare_class_OHE_vector(real_cat=real_cat, num_classes=self.C) # prepare ohe vector
            annot[0] = 1. # required by yolo, means the probability, that an object is detected
            annot.extend(ohe_vector) # add ohe vector to the end of annot
            annot.extend([ind_x, ind_y]) # append grid cell coords at the end
        return annots
    
    def _global_relative_to_absolute(self, coords: tuple[float], img_size: tuple[int]) -> tuple[float]:

        return (int(coords[0]*img_size[0]), int(coords[1]*img_size[1]))
    
    def _global_absolute_to_local_relative(self, coords: tuple[float], img_size: tuple[int], S:int):
        # calcualtes the relative postion of (xc, yc) with respect to the grid cell UL corner
        
        # get grid cell size (in pixels)
        dx = img_size[0] // S
        dy = img_size[1] // S

        # get relative xc, yc with respect to grid cell UL corner
        rel_x = (coords[0] % dx) / dx 
        rel_y = (coords[1] % dy) / dy 

        # get grid cell number (index_x, index_y) - each from 0 to S-1
        grid_x = (coords[0] // dx) 
        grid_y = (coords[1] // dy)

        return rel_x, rel_y, grid_x, grid_y
    
    def _prepare_class_OHE_vector(self, real_cat:int, num_classes:int) -> list:

        assert real_cat <= num_classes, "selected category index exceeds number of classes" # make sure, that selected class is within allowed classes

        # prepare one hot encoded vector
        ohe: list = [0.] *num_classes
        ohe[real_cat] = 1.

        return ohe

