import argparse
import yaml
import torch
from train import create_dataloader
from train import load_model
from train import validate

def main():
    parser = argparse.ArgumentParser(description='Evaluate deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    parser.add_argument('--model', help='Path to best model .pt')
    parser.add_argument('--split', help="Define split", default="val")
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for validation set
    dl_val = create_dataloader(cfg, split=args.split)

    # initialize model
    model_instance, _ = load_model(cfg, model_path=args.model)

    # metrics
    loss_val, oa_val, precision, recall, average_precision, prec_recall_curve = validate(cfg, dl_val, model_instance)

    print(loss_val, oa_val, precision, recall, average_precision, prec_recall_curve)

if __name__ == '__main__':
    main()
