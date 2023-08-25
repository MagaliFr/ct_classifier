'''
    Testing script. Here, we load the trained model and export predicted
    values, true values, and filepaths of images as .json.

'''
import os
import argparse
import yaml
import glob
#from tqdm import trange
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
import numpy as np
from train import create_dataloader


# define function to load model
def load_model(cfg, model_path):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class
    overwrite = cfg['overwrite']

    state = torch.load(open(model_path, 'rb'), map_location='cpu')
    model_instance.load_state_dict(state['model'])

    return model_instance


# define test function (where the true labels, predicted labels, and filepaths are logged)
def test(cfg, model, dataLoader):
    '''
        Function for selecting a subset of images to be logged into Comet. 
        Images taken are...?
    '''
    
    device = cfg['device']
    model.to(device)

    # put model into eval mode
    model.eval()

    # create empty lists for true labels,  predicted labels, and image filepaths
    true_labels = []
    pred_labels = []
    logits = []
    scores = []
    filepaths = []
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels, image_paths) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # add true labels to the true labels list
            #import pdb; pdb.set_trace() # DEBUGGER
            labels_np = labels.cpu().detach().numpy()
            true_labels.extend(labels_np)

            # forward pass
            prediction = model(data)

            logits.extend(prediction.cpu().detach().numpy().tolist())
            # raise ValueError(logits)
            scores.extend(prediction.softmax(dim=1).cpu().detach().numpy().tolist())
            
            # add predicted labels to the predicted labels list
            pred_label = torch.argmax(prediction, dim=1)
            pred_label_np = pred_label.cpu().detach().numpy()
            pred_labels.extend(pred_label_np)

            filepaths.extend(image_paths)

            # print(len(pred_labels), len(true_labels), logits[-1], scores[-1], len(filepaths))

    return pred_labels, true_labels, logits, scores, filepaths


def main():
    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Test deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='')
    parser.add_argument('--model_path', help='Path to model .pt file', default='')
    parser.add_argument('--save_json_path', help='Path to .json output', default='')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg['batch_size'] = 2048

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    dl_val = create_dataloader(cfg, split='val')

    # initialize model
    model = load_model(cfg, args.model_path)

    # metrics
    pred_labels, true_labels, logits, scores, filepaths = test(cfg, model, dl_val)

    output_dict = {'pred_labels': [int(i) for i in pred_labels], 
                   'true_labels': [int(i) for i in true_labels],
                   'logits':logits,
                   'scores':scores,
                   'filepaths':filepaths}

    output_dict.keys()
    output_dict['pred_labels'][0]
    # import pdb; pdb.set_trace()
    # save as a json
    with open(args.save_json_path,'w') as f:
        json.dump(output_dict, f)


# to call the test.py script directly from cmd:
if __name__ == '__main__':
    main()