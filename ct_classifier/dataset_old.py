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

    #def _get_transform_pad_resize(target_size):
    #    ''' Returns a transform to pad and resize the image to the target size. '''
    #    return T.Compose([
    #        CTDataset._calculate_padding,  # Pad first
    #        T.Resize(target_size),  # Then resize
    #        T.ToTensor()
    #    ])
    
    #def _get_transform_pad_resize(target_size):
    #    ''' Returns a transform to pad and resize the image to the target size. '''
    #    return T.Compose([
    #    T.Lambda(lambda img: img.pad(CTDataset._calculate_padding(img))),  # Use a lambda to apply the padding
    #    T.Resize(target_size),
    #    T.ToTensor()
    #    ])

    #def _get_transform_pad_resize(target_size):
    #    ''' Returns a transform to pad and resize the image to the target size. '''
    #    return T.Compose([
    #    T.Pad(CTDataset._calculate_padding),  # Use Pad directly
    #    T.Resize(target_size),
    #    T.ToTensor()
    #    ])

    #def _calculate_padding(img):
    #    w, h = img.size
    #    img_max_dim = max(w, h)  # Which size is longer?
    #    pad_amount_x = int((img_max_dim - w) / 2)
    #    pad_amount_y = int((img_max_dim - h) / 2)
    #    return (pad_amount_x, pad_amount_y, pad_amount_x, pad_amount_y)  # Returns a 4-tuple (left, top, right, bottom)



    #def _calculate_padding(img):
    #    w, h = img.size
    #   img_max_dim = max(w, h)  # Which size is longer?
    #    pad_amount_x = int((img_max_dim - w) / 2)
    #    pad_amount_y = int((img_max_dim - h) / 2)
    #    return [pad_amount_x, pad_amount_y]
    

    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = image_name# os.path.join(self.data_root, 'PrototypeCroppedImages/PrototypeCroppedImages_Age_Test', image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # calculate padding for current image
        #w, h = img.size
        #img_max_dim = max(w, h) # which size is longer?
        #pad_amount_x = int((img_max_dim - w) / 2)
        #pad_amount_y = int((img_max_dim - h) / 2)
        #transform_pad_resize = T.Compose([
        #    T.Pad([pad_amount_x, pad_amount_y]), # pad first
        #    T.Resize(self.image_size), # then resize
        #    T.ToTensor()
        #])
        #img = Image.open(image_path).convert('RGB')
        #img = transform_and_pad(img, self.image_size)
        #img = Image.open(image_path).convert('RGB')
        #transform_pad_resize = CTDataset._get_transform_pad_resize(self.image_size)
        img_tensor = transform_pad_resize(img)

        # Calculate the padding
        w, h = img.size
        img_max_dim = max(w, h)
        pad_amount_x = int((img_max_dim - w) / 2)
        pad_amount_y = int((img_max_dim - h) / 2)

        # Apply padding
        img = T.Pad([pad_amount_x, pad_amount_y])(img)

        # Apply the resize and ToTensor transformations
        img = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor()
        ])(img)

        # Apply the transformation with padding
        #img_tensor = transform_pad_resize(img)


        # transform: see lines 31ff above where we define our transformations
        # img_tensor = self.transform(img)

        return img_tensor, img, label, image_path
    

def visualize_one_image(img_path, target_size):
    img = Image.open(img_path).convert('RGB')
    
    # Calculate the padding
    w, h = img.size
    img_max_dim = max(w, h)  # Which size is longer?
    pad_amount_x = int((img_max_dim - w) / 2)
    pad_amount_y = int((img_max_dim - h) / 2)
    
    transform_pad_resize = CTDataset._get_transform_pad_resize(target_size)
    
    padded_img = transform_pad_resize(img)
    
    # Convert the padded tensor back to an image
    padded_img_from_tensor = T.ToPILImage()(padded_img.squeeze(0).cpu())
    plt.imshow(padded_img_from_tensor)
    plt.title('Padded Image')
    plt.show()

