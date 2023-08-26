'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model 
#from comet_ml.integration.pytorch import log_image
from PIL import Image, ImageDraw, ImageFont

import os
import argparse
import yaml
import glob
from tqdm import trange
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import json
from torchvision import transforms
import torchvision.datasets as datasets
from sklearn.metrics import precision_recall_curve
from torch import softmax as softmax
#from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from sklearn.metrics import precision_score, recall_score

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18

experiment = Experiment(
    api_key="74VvHHBrA7mvKzWBsmvNV1nqf",
    project_name="general",
    workspace="magalifr"
    )

# for logging best model through epochs
def save_model(model, filename):
    torch.save(model.state_dict(), filename)



def log_images_from_coco_json(json_path, folder_prefix, num_images_to_log=10):
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    image_paths = [item["file_name"] for item in coco_data["images"]]
    
    for path in image_paths[:num_images_to_log]:
        experiment.log_image(path, name=f"{folder_prefix}/{path.split('/')[-1]}")


def overlay_labels_on_image(image_path, ground_truth, prediction):
    # Open an image file
    image = Image.open(image_path)
    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Specify font size and color
    font = ImageFont.load_default() 
    #font = ImageFont.truetype("Ubuntu-R.ttf", size=25)  # you might need a different font file
    color = "red"
    
    # Draw ground truth and prediction on the image
    draw.text((10, 10), f"Ground Truth: {ground_truth}", fill=color, font=font)
    draw.text((10, 40), f"Prediction: {prediction}", fill=color, font=font)

    # Save the modified image or return it
    # image.save("output_path.jpg")  # if you want to save
    return image


def draw_label_on_image(image, label, position=(10, 10)):
    """
    Draw a label on an image using PIL.

    image: PIL Image object.
    label: Text to display on the image.
    position: Tuple indicating where to start the text.

    Returns a new image with the label.
    """
    # If the image is not a PIL Image, convert it
    if not isinstance(image, Image.Image):
        image = transforms.ToPILImage()(image)

    draw = ImageDraw.Draw(image)
    
    # Use a basic font. For custom fonts, you might need to use ImageFont.truetype
    font = ImageFont.load_default()

    draw.text(position, label, (255, 255, 255), font=font)
    return image


def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True if split == 'train' else False,
            num_workers=cfg['num_workers']
        )
    return dataLoader



def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class
    overwrite = cfg['overwrite']

    # load latest model state
    model_states = glob.glob(os.path.join(cfg['save_dir'],'*.pt')) #'model_states/*.pt') #log_images_from_coco_json(os.path.join(cfg['data_root'], json_file_val), "validation")
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace(cfg['save_dir'],'').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(os.path.join(cfg['save_dir'], f'{start_epoch}.pt'), 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch



def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs(cfg['save_dir'], exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    #torch.save(stats, open(os.path.join(cfg['save_dir']),f'{epoch}.pt', 'wb'))
    torch.save(stats, open(os.path.join(cfg['save_dir'], f'{epoch}.pt'), 'wb'))
    
    # also save config file if not present
    cfpath = os.path.join(cfg['save_dir'],'config.yaml')
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

            

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''
    #dataset_instance = CTDataset(cfg, split) 

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss()

    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    #logging_interval = 10  # Log every 10 batches, adjust this value as per your needs
    #num_images_to_log = 29
    for idx, (data, labels, img_path) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

    #    for j, image_path in enumerate():  
    #        modified_image = overlay_labels_on_image(image_path, labels[j], prediction[j])
    #        experiment.log_image(modified_image, name=f"batch_{i}_img_{j}.jpg")

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        #print(prediction, labels)
        #loss = torch.clamp(loss, min=0, max=10)
        #print(loss)
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        #prediction_softmax = softmax(prediction)
        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
           )
        )
        progressBar.update(1)

        #for i in range(min(data.size(0), num_images_to_log)):  # log a few images from the batch
        #    image = data[i].permute(1, 2, 0).cpu().numpy()  # permute and convert to numpy
        #    labels = labels[i].item()
        #    pred_label = pred_label[i].argmax(dim=0).item()  # get the predicted class

            # Log the image with label and prediction to Comet.ml here
            #...
            # For the purpose of visualization, take the first image from the batch
        #    image = data[0]
        #    ground_truth_label = labels[0].item()

            # Get the model's prediction. This is just a basic example, adapt it to your case.
        #    prediction = model(data).argmax(dim=1)[0].item()

        #    labeled_image = draw_label_on_image(image, f"GT: {ground_truth_label} | Pred: {prediction}")
    
            # Log the image with labels to Comet ML
        #    experiment.log_image(labeled_image, name=f"batch_{idx}_image_0.jpg")

    # You might not want to log every image in every batch, so consider adding a condition to log periodically
       # if idx % logging_interval == 0:
       #     break

            # If you want to log images, predictions, or overlay ground truth on images:
        #for j, image_path in enumerate(image_paths):
        # Logic for processing each image in the batch using image_path
        # For example, overlay and log images with predictions:
        #    overlayed_image = overlay_labels_on_image(image_path, labels[j], prediction[j])
        #    experiment.log_image(overlayed_image, name=f"batch_{idx}_img_{j}.jpg")

    #log_images_from_coco_json("/home/magali/CV4Ecology-summer-school/FinalDataset/SubsetAgeModelCocoTrain_croppedID.json", "training")
    json_file_train = cfg['json_file_train']
    log_images_from_coco_json(os.path.join(cfg['data_root'], json_file_train), "training")

    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total


def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    # create empty lists for true and predicted labels
    true_labels = []
    pred_labels = []
    all_labels = []
    all_pred_labels = []

    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels, img_path) in enumerate(dataLoader):
            
            all_labels = all_labels + labels.tolist()

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # add true labels to the true labels list
            # import pdb; pdb.set_trace() # DEBUGGER
            labels_np = labels.cpu().detach().numpy()
            true_labels.extend(labels_np)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            #prediction_softmax = softmax(prediction)
            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            all_pred_labels = all_pred_labels + pred_label.cpu().tolist()

            # add predicted labels to the predicted labels list
            pred_label_np = pred_label.cpu().detach().numpy()
            pred_labels.extend(pred_label_np)

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    json_file_val = cfg['json_file_val']
    log_images_from_coco_json(os.path.join(cfg['data_root'], json_file_val), "validation")
    #log_images_from_coco_json(os.path.join"/home/magali/CV4Ecology-summer-school/FinalDataset/SubsetAgeModelCocoVal_croppedID.json", "validation")

    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    # calculate precision and recall
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

    #experiment.log_metric("loss val", loss_total)
    #experiment.log_metric("acc val", oa_total)

    # confusion matrix
    #experiment.create_confusion_matrix(y_true=labels, y_predicted=pred_label)
    experiment.log_confusion_matrix(y_true=all_labels, y_predicted=all_pred_labels)

    #precision, recall, thresholds = precision_recall_curve(labels, preds)
    #precision_recall_curve(labels,prediction)

    # print nr of lables and predictions
    print('all_labels',len(all_labels), 'all_pred', len(all_pred_labels))

    return loss_total, oa_total, precision, recall



def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val, precision, recall = validate(cfg, dl_val, model)

        # save best model
        best_acc = 0.0
        best_epoch = -1
        if oa_val > best_acc:
            best_acc = oa_val
            best_epoch = current_epoch
        

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            'precision' : precision,
            'recall' : recall
        }

        experiment.log_metrics(stats, epoch = current_epoch)
        #log_metrics(dic, prefix=None, step=None, epoch=None)

        save_model(cfg, current_epoch, model, stats)  

    save_model(model, "best_model.pt")  

    # That's all, folks!
        
if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    # Paths to your COCO JSON files
    #training_json_path = "/FinalDataset/SubsetAgeModelCocoTrain_croppedID.json" #path_to_training_coco.json
    #validation_json_path = "/FinalDataset/SubsetAgeModelCocoVal_croppedID.json" #path_to_validation_coco.json
    #testing_json_path = "/FinalDataset/SubsetAgeModelCocoTest_croppedID.json" #path_to_testing_coco.json

    # Initialize your Comet.ml experiment
    # experiment = Experiment(api_key="your_api_key", project_name="general", workspace="your_workspace")

    # Call your functions with the desired paths
    #train(training_json_path)
    #val(validation_json_path)
    #test(testing_json_path)
    main()
