import torch
import yaml

from train import load_model
from train import create_dataloader

cfg = yaml.safe_load(open('configs/exp_resnet18.yaml', 'r'))
model, current_epoch = load_model(cfg)
#model = torch.load('/home/magali/ct_classifier/model_states/48.pt')['model']
print(type(model))

dl_val = create_dataloader(cfg, split='val')

device = cfg['device']
model.to(device)

model.eval()

with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dl_val):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            print(data.shape)
            #print(model.classifier.device)
            print(data.device)
            print(device)
            # forward pass
            prediction = model(data)
            print(prediction)
            break

print(data[0].shape)