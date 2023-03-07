import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import pdb
import numpy as np
from flopth import flopth
from ofa.imagenet_classification.elastic_nn.networks.ofa_resnets import OFAResNets
from sklearn.metrics import f1_score,confusion_matrix

DATA_PATH = 'HC701/ass2/TBX11K/imgs'
CSV_PATH = 'task21.csv'

dataframe = pd.read_csv(CSV_PATH)
test_data = dataframe[dataframe['types']=='test']['filename'].tolist()
test_labels = dataframe[dataframe['types']=='test']['labels'].tolist()
train_data = dataframe[dataframe['types']=='train']['filename'].tolist()
train_labels = dataframe[dataframe['types']=='train']['labels'].tolist()
# tb 1 health 0
# to tensor
train_labels = torch.tensor([1 if label == 'tb' else 0 for label in train_labels])
test_labels = torch.tensor([1 if label == 'tb' else 0 for label in test_labels])



def train(model,train_loader,criterion,optimizer,save_path,epoch):
    model.train()
    train_loss = 0
    total = 0
    correct=0
    for batch_idx, (data, target) in enumerate(train_loader):
        # send to device
        data, target = data.float().to(device), target.long().to(device)
        pdb.set_trace()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    print(' train loss: {:.4f} accuracy: {:.4f}'.format(train_loss/(batch_idx+1), 100.*correct/total))
    # wandb.log({"train_loss": train_loss/(batch_idx+1),"train_accuracy":100.*correct/total,'epoch':epoch})
    # save model
    torch.save(model.state_dict(), os.path.join(save_path,f"epoch{epoch}.pt"))
            


def validate(model,val_loader,epoch):   
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    preds = []
    gts = []
    for batch_idx, (data, target) in enumerate(val_loader):
        # send to device
        data, target = data.float().to(device), target.long().to(device)
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        test_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        preds += predicted.cpu().tolist()
        gts += target.cpu().tolist()
    f1 = f1_score(gts,preds,average='weighted')
    cf_matrix = confusion_matrix(gts,preds)
    print(' test loss: {:.4f} accuracy: {:.4f}'.format(test_loss/(batch_idx+1), 100.*correct/total))
    print(f'cf_matrix {cf_matrix}')
    # wandb.log({"test_accuracy": 100.*correct / total,"test_loss":test_loss/(batch_idx+1),'f1_score':f1,'epoch':epoch})
    
# data augmentation
class MyDataset(Dataset):
    def __init__(self,data,label) -> None:
        super().__init__()
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x,y



def prepare_dataset(dataset,labels,transform=None):
    data_img = []
    for file in dataset:
        if file.startswith('h'):
            img = Image.open(os.path.join(DATA_PATH,'health',file))
        elif file.startswith('t'):
            img = Image.open(os.path.join(DATA_PATH,'tb',file))
        # transform
        if transform:
            img = transform(img)
        data_img.append(img[None,])
    return torch.cat(data_img), np.array(labels).squeeze()


if __name__ == '__main__':
    
    wandb.init(project="hc_assign2_exp5", config={"learning_rate": 0.0001})
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    p_train_data,p_train_label = prepare_dataset(train_data,train_labels,transform)
    p_test_data,p_test_label= prepare_dataset(test_data,test_labels,transform=transforms.ToTensor())
    train_dataset = MyDataset(p_train_data,p_train_label)
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    test_dataset = MyDataset(p_test_data,p_test_label)
    test_loader = DataLoader(test_dataset,batch_size=4)
    print('finish prepare data')
    model = OFAResNets(n_classes=2,dropout_rate=0.5, depth_list=[0, 1], expand_ratio_list=[0.2, 0.3], width_mult_list=[0.8, 0.8])
    # print model parameters
    pytorch_total_params = sum(p.numel() for p in  model.parameters())
    print('Number of parameters: {0}'.format(pytorch_total_params))
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # print flops
    flops, params = flopth(model, in_size=((3, 512, 512),))
    print(f'flops {flops}')
    print("start training")
    # train
    epochs = 20
    save_path = 'HC701/ass2/result_exp5'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        train(model,train_loader,criterion,optimizer,save_path,epoch)
        # val
        validate(model,test_loader,epoch)

    wandb.finish()       
    

    
   