from argparse import ArgumentParser

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import CNNClassifier, SWAG
from src.trainer import Trainer

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_epochs', type=int, default=5)
    # swag params
    parser.add_argument('--swa_start', type=int, default=2)
    parser.add_argument('--swa_c_epochs', type=int, default=1)
    parser.add_argument('--cov_mat', action='store_true')
    parser.add_argument('--max_num_models', type=int, default=10)
    
    return parser.parse_args()

def main():
    # args
    args = get_args()
    # dataloader
    train_loader = DataLoader(datasets.MNIST('../mnist_data', 
                                            download=True,
                                            train=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                ])), 
                                            batch_size=64, 
                                            shuffle=True)

    # download and transform test dataset
    test_loader = DataLoader(datasets.MNIST('../mnist_data',
                                            download=True,
                                            train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                ])), 
                                            batch_size=64, 
                                            shuffle=True)
    loaders = [train_loader, test_loader]
    # model
    model = CNNClassifier()
    # swag model
    swag_model = SWAG(CNNClassifier, no_cov_mat=not(args.cov_mat), max_num_models=args.max_num_models)
    # trainer
    trainer = Trainer(args, model, swag_model, loaders)
    trainer.train()

if __name__ == '__main__':
    main()