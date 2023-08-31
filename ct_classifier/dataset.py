'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt


class CTDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])

        # save image size from cfg
        self.image_size = cfg['image_size']
        
        # index data into list
        self.data = []

        json_file_train = cfg['json_file_train']
        json_file_val = cfg['json_file_val']


        # load annotation file
        annoPath = os.path.join(
            self.data_root,
            #'eccv_18_annotation_files',
            json_file_train if self.split=='train' else json_file_val
        )
        meta = json.load(open(annoPath, 'r'))

        images = dict([[i['id'], i['file_name']] for i in meta['images']])          # image id to filename lookup
        labels = dict([[c['id'], idx] for idx, c in enumerate(meta['categories'])]) # custom labelclass indices that start at zero
        
        # since we're doing classification, we're just taking the first annotation per image and drop the rest
        images_covered = set()      # all those images for which we have already assigned a label
        for anno in meta['annotations']:
            imgID = anno['image_id']
            if imgID in images_covered:
                continue
            
            # append image-label tuple to data
            imgFileName = images[imgID]
            label = anno['category_id']
            labelIndex = labels[label]
            self.data.append([imgFileName, labelIndex])
            images_covered.add(imgID)       # make sure image is only added once to dataset
    
        print(split, 'number of images', len(images), 'labels', len(self.data), 'images covered', len(images_covered))


    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = image_name# os.path.join(self.data_root, 'PrototypeCroppedImages/PrototypeCroppedImages_Age_Test', image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # calculate padding for current image & adding augmentations (outcomment if basic model run)
        w, h = img.size
        img_max_dim = max(w, h) # which size is longer?
        pad_amount_x = int((img_max_dim - w) / 2)
        pad_amount_y = int((img_max_dim - h) / 2)
        if self.split == "train":
            transforms = T.Compose([
                T.Pad([pad_amount_x, pad_amount_y]), # pad first
                T.Resize(self.image_size), # then resize
                #T.RandomHorizontalFlip(p=0.5),
                #T.ColorJitter(brightness=.5, hue=.3),
                T.ToTensor()
            ])
        else:
            transforms = T.Compose([
                T.Pad([pad_amount_x, pad_amount_y]), # pad first
                T.Resize(self.image_size), # then resize
                T.ToTensor()
            ])
        # Apply the transformation with padding
        img_tensor = transforms(img)

        # transform: see lines 31ff above where we define our transformations
        #img_tensor = self.transform(img) #activate this if only resize without padding

        return img_tensor, label, image_path