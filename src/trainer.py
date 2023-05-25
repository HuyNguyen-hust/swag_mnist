import torch
from torch.optim import SGD
import torch.nn.functional as F
from . import utils

class Trainer:
    def __init__(self, args, model, swag_model, loaders):
        self.args = args
        self.model = model
        self.swag_model = swag_model
        self.train_loader = loaders[0]
        self.test_loader = loaders[1]
        params = [p for p in model.parameters() if p.requires_grad == True]
        self.optim = SGD(params, lr=0.01, momentum=0.5)
        
        # device
        self.model.to(args.device)
        self.swag_model.to(args.device)
        
        # some args
        self.init_training()
        
    def init_training(self):
        self.sgd_ens_preds = None
        self.sgd_targets = None
        self.n_ensembled = 0.0
        self.loss_history = []
        self.acc_history = []

    def train(self):
        for i in range(self.args.num_epochs):
            self.train_one_epoch(i)
            self.eval()
            if ((i + 1) > self.args.swa_start  and (i + 1 - self.args.swa_start) % self.args.swa_c_epochs == 0):
                self.swa_step()
            print('-' * 30)
        
    def train_one_epoch(self, i):
        batch_size = 64
        for batch_id, (data, label) in enumerate(self.train_loader):
            data = data.to(self.args.device)
            target = label.to(self.args.device)
            
            # forward pass, calculate loss and backprop!
            self.optim.zero_grad()
            preds = self.model(data)
            loss = F.nll_loss(preds, target)
            loss.backward()
            self.loss_history.append(loss.item())
            self.optim.step()
        
            if batch_id % 100 == 0:
                print('epoch {} - step {} | loss {}'.format(i, batch_id, loss.item()))
    
    def eval(self):
        self.model.eval() # set model in inference mode (need this because of dropout)
        with torch.no_grad():
            test_res = utils.eval(self.test_loader, self.model, cross_entropy, cuda=True)
            print('-----test results-----')
            print(test_res)
            
    def swa_step(self):
        self.model.eval() # set model in inference mode (need this because of dropout)
        
        with torch.no_grad():
            sgd_res = utils.predict(self.test_loader, self.model)
            sgd_preds = sgd_res["predictions"]
            sgd_targets = sgd_res["targets"]
                
        if self.sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            # TODO: rewrite in a numerically stable way
            sgd_ens_preds = sgd_ens_preds * self.n_ensembled / (
                self.n_ensembled + 1
            ) + sgd_preds / (self.n_ensembled + 1)
        self.n_ensembled += 1
        self.swag_model.collect_model(self.model)
        self.swag_model.sample(0.0)
        utils.bn_update(self.train_loader, self.swag_model)
        print('-----swag test results-----')
        swag_res = utils.eval(self.test_loader, self.swag_model, cross_entropy)
        print(swag_res)


def cross_entropy(model, input, target):
    # standard cross-entropy loss function

    output = model(input)

    loss = F.cross_entropy(output, target)

    return loss, output