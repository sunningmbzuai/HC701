# %% [markdown]
# ## Assignment3

# %%
import wandb
import numpy as np 
import pandas as pd
# from torchsummary import summary
    
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
import pdb
from sklearn.metrics import log_loss as CE_loss

# %%
### preprocess data

train_csv = "/apps/local/shared/HC701/assessment/assignment_3/data/hc701_lits_train.csv"
test_csv = "/apps/local/shared/HC701/assessment/assignment_3/data/hc701_lits_test.csv"
data_root = "/apps/local/shared/HC701/assessment/assignment_3/data"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# only load data with both mask
train_data = train_df[(train_df['tumor_mask_empty']==True) & (train_df['liver_mask_empty']==True)]
test_data = test_df[(test_df['tumor_mask_empty']==True) & (test_df['liver_mask_empty']==True)]


# %%
class ConsecutiveConvolution(nn.Module):
    def __init__(self,input_channel,out_channel):
        super(ConsecutiveConvolution,self).__init__()
        self.conv = nn.Sequential(
            
            nn.Conv2d(input_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),            
        
        )
        
    def forward(self,x):
        return self.conv(x)

# %%
class UNet(nn.Module):
    def __init__(self,input_channel, output_channel, features = [64,128,256,512]):
        super(UNet,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # initialize the encoder
        for feat in features:
            self.encoder.append(
                ConsecutiveConvolution(input_channel, feat)    
            )
            input_channel = feat
        
        #initialize the decoder 
        for feat in reversed(features):
            # the authors used transpose convolution
            self.decoder.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2))
            self.decoder.append(ConsecutiveConvolution(feat*2, feat))
        
        #bottleneck
        self.bottleneck = ConsecutiveConvolution(features[-1],features[-1]*2)
        
        #output layer
        self.final_layer = nn.Conv2d(features[0],output_channel,kernel_size=1)
        
    def forward(self,x):
        skip_connections = []
        
        #encoding
        for layers in self.encoder:
            x = layers(x)
            #skip connection to be used in recreation 
            skip_connections.append(x)

            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        
        for idx in range(0,len(self.decoder),2):
            
            
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            
    
            if x.shape != skip_connection.shape[2:]:
                x = TF.resize(x,size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection,x),dim=1)
#             print(concat_skip.shape)
#             print(self.decoder[idx+1])

            x = self.decoder[idx+1](concat_skip)
        
        return self.final_layer(x)
            

# %%
from PIL import Image
from torch.utils.data import Dataset

class CXRDataset(Dataset):
    def __init__(self, image_list, liver_mask_list,tumor_mask_list,img_root,split=None,transform=None):
        self.image_list = image_list
        self.liver_mask_list = liver_mask_list
        self.tumor_mask_list = tumor_mask_list
        self.transform = transform
        self.img_root = img_root
        # split validate dataset from train dataset
        if split == 'train':
            total_len = len(self.image_list)
            self.image_list = self.image_list[:int(0.8*total_len)]
            self.liver_mask_list = self.liver_mask_list[:int(0.8*total_len)]
            self.tumor_mask_list = self.tumor_mask_list[:int(0.8*total_len)]
        elif split == 'val':
            total_len = len(self.image_list)
            self.image_list = self.image_list[int(0.8*total_len):]
            self.liver_mask_list = self.liver_mask_list[int(0.8*total_len):]
            self.tumor_mask_list = self.tumor_mask_list[int(0.8*total_len):]


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        liver_mask_path = os.path.join(self.img_root, self.liver_mask_list[index])
        tumor_mask_path = os.path.join(self.img_root, self.tumor_mask_list[index])
        img_path = os.path.join(self.img_root, self.image_list[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        liver_mask = np.array(Image.open(liver_mask_path).convert("L"), dtype=np.float32)
        tumor_mask = np.array(Image.open(tumor_mask_path).convert("L"), dtype=np.float32)
        # liver_mask[liver_mask == 255.0] = 1.0
        # tumor_mask[tumor_mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, liver_mask=liver_mask, tumor_mask=tumor_mask)
            image = augmentations["image"]
            liver_mask = augmentations["liver_mask"].squeeze()
            tumor_mask = augmentations["tumor_mask"].squeeze()

        return image, liver_mask, tumor_mask

# %%
import torchvision
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_data,
    test_data,
    data_root,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,):
    
    
    train_ds = CXRDataset(
        image_list=train_data['filepath'].tolist(), 
        liver_mask_list=train_data['liver_maskpath'].tolist(),
        tumor_mask_list=train_data['tumor_maskpath'].tolist(),
        img_root=data_root,
        split='train',
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CXRDataset(
        image_list=train_data['filepath'].tolist(), 
        liver_mask_list=train_data['liver_maskpath'].tolist(),
        tumor_mask_list=train_data['tumor_maskpath'].tolist(),
        img_root=data_root,
        split='val',
        transform=train_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    test_ds = CXRDataset(
        image_list=test_data['filepath'].tolist(), 
        liver_mask_list=test_data['liver_maskpath'].tolist(),
        tumor_mask_list=test_data['tumor_maskpath'].tolist(),
        img_root=data_root,
        transform=val_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def IOU(mask1: torch.Tensor, mask2: torch.Tensor):
    intersection = (mask1 * mask2).sum()
    union = (mask1 + mask2).sum() - intersection
    return intersection / union

def val_fn(loader, model, device="cuda",epoch=0):
    liver_num_correct = 0
    tumor_num_correct = 0
    liver_num_pixels = 0
    tumor_num_pixels = 0
    liver_dice_score = 0
    tumor_dice_score=0
    liver_iou = 0
    tumor_iou = 0
    liver_ce = 0
    tumor_ce = 0
    model.eval()

    with torch.no_grad():
        for x, y1,y2 in loader:
            x = x.to(device)
            y1 = y1.to(device).unsqueeze(1)
            y2 = y2.to(device).unsqueeze(1)
            preds = model(x)
            preds1 = (torch.sigmoid(preds[:,0].unsqueeze(1)) > 0.5).float()
            preds2 = (torch.sigmoid(preds[:,1].unsqueeze(1)) > 0.5).float()
            liver_num_correct += (preds1 == y1).sum()
            tumor_num_correct += (preds2 == y2).sum()
            liver_num_pixels += torch.numel(preds1)
            tumor_num_pixels += torch.numel(preds2)
            liver_dice_score += (2 * (preds1 * y1).sum()) / (
                (preds1 + y1).sum() + 1e-8
            )
            tumor_dice_score += (2 * (preds2 * y2).sum()) / (
                (preds2 + y2).sum() + 1e-8
            )
            
            liver_iou += IOU(preds1.type(torch.int64),y1.type(torch.int64))
            tumor_iou += IOU(preds2.type(torch.int64),y2.type(torch.int64))
           

    wandb.log({"val liver acc": liver_num_correct/liver_num_pixels*100,
               "val liver Dice":liver_dice_score/len(loader),
               "val tumor acc":tumor_num_correct/tumor_num_pixels*100,
               "val tumor Dice":tumor_dice_score/len(loader),
               "val liver IOU": liver_iou/len(loader),
               "val tumor IOU": tumor_iou/len(loader),
               'epoch':epoch})
    

# def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    
#     model.eval()
#     for idx, (x, y1, y2) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds1,preds2 = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

#     model.train()

def test_fn(loader, model, device="cuda",epoch=0):
    model.eval()
    liver_num_correct = 0
    tumor_num_correct = 0
    liver_num_pixels = 0
    tumor_num_pixels = 0
    liver_dice_score = []
    tumor_dice_score=[]
    liver_iou = []
    tumor_iou = []
    with torch.no_grad():
        for x, y1,y2 in loader:
            x = x.to(device)
            y1 = y1.to(device).unsqueeze(1)
            y2 = y2.to(device).unsqueeze(1)
            preds = model(x)
            preds1 = (torch.sigmoid(preds[:,0].unsqueeze(1)) > 0.5).float()
            preds2 = (torch.sigmoid(preds[:,1].unsqueeze(1)) > 0.5).float()
            liver_num_correct += (preds1 == y1).sum()
            tumor_num_correct += (preds2 == y2).sum()
            liver_num_pixels += torch.numel(preds1)
            tumor_num_pixels += torch.numel(preds2)
            liver_dice_score.append(((2 * (preds1 * y1).sum()) / (
                (preds1 + y1).sum() + 1e-8)).cpu()
            )
            tumor_dice_score.append( ((2 * (preds2 * y2).sum()) / (
                (preds2 + y2).sum() + 1e-8)).cpu()
            )
            liver_iou.append(IOU(preds1.type(torch.int64),y1.type(torch.int64)).cpu())
            tumor_iou.append(IOU(preds2.type(torch.int64),y2.type(torch.int64)).cpu())

    wandb.log({"test liver Dice mean":np.mean(np.array(liver_dice_score)),
               "test liver Dice std": np.std(np.array(liver_dice_score)),
               "test tumor Dice mean":np.mean(np.array(tumor_dice_score)),
               "test tumor Dice std": np.std(np.array(tumor_dice_score)),
               "test liver Jaccard mean": np.mean(np.array(liver_iou)),
               "test liver Jaccard std":np.std(np.array(liver_iou)),
               "test tumor Jaccard mean": np.mean(tumor_iou),
               "test tumor Jaccard std": np.std(tumor_iou),
               'epoch':epoch})
  
    

# %%
# hyperparams
lr = 1e-4
dev = "cuda"
batch_size = 8
epochs = 50
workers= 8
img_h = 256
img_w = 256
pin_mem= True
load_model = False


def focal_loss(mask_pred, mask_true, alpha=0.25, gamma=2):
    """
    Compute Focal Loss for binary segmentation masks.

    Args:
        mask_pred (torch.Tensor): predicted segmentation mask of shape (batch_size, 1, height, width)
        mask_true (torch.Tensor): ground truth segmentation mask of shape (batch_size, 1, height, width)
        alpha (float): balancing factor between positive and negative samples. Default: 0.25
        gamma (float): focusing parameter for modulating the weight assigned to each pixel. Default: 2

    Returns:
        focal_loss (torch.Tensor): scalar tensor of focal loss value averaged across batch.
    """
    # Flatten both the predicted and ground truth masks
    bs = mask_pred.shape[0]
    mask_pred = mask_pred.reshape(bs,-1)
    mask_true = mask_true.reshape(bs,-1)

    # Compute binary cross entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_true, reduction='none')

    # Compute the modulating factor for focal loss
    modulating_factor = torch.exp(-gamma * mask_true * mask_pred - gamma * torch.log(1 + torch.exp(-mask_pred)))

    # Compute the final focal loss
    focal_loss = (alpha * (1 - mask_pred)**gamma * bce_loss + (1 - alpha) * mask_pred**gamma * modulating_factor * bce_loss).mean()

    return focal_loss
# %%
def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    train_loss = 0.0
    for batch_idx, (data, targets1,targets2) in enumerate(loop):
        data = data.to(device=dev)
        targets1 = targets1.float().to(device=dev)
        targets2 = targets2.float().to(device=dev)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss1 = loss_fn(predictions[:,0], targets1)
            loss2 = loss_fn(predictions[:,1], targets2)
            loss3 = focal_loss(predictions[:,0], targets1)
            loss4 = focal_loss(predictions[:,1], targets2)
            loss = loss1 + loss2 + loss3 + loss4

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    wandb.log({"train_loss": train_loss/(batch_idx+1)})

# %%
train_transform = A.Compose(
        [
            A.Resize(height=img_h, width=img_w),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

val_transforms = A.Compose(
    [
        A.Resize(height=img_h, width=img_w),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
## model
model = UNet(input_channel=3, output_channel=2).to(dev)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_loader, val_loader,test_loader = get_loaders(
    train_data,
    test_data,
    data_root,
    batch_size,
    train_transform,
    val_transforms,
    workers,
    pin_mem,
)

# if LOAD_MODEL:
#     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
exp_num = 5
if not os.path.exists(f'exp{exp_num}'):
    os.mkdir(f'exp{exp_num}')
    
wandb.init(project=f"hc_assign3_exp{exp_num}", config={"learning_rate":lr})

scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    print(epoch)
    train_fn(train_loader, model, optimizer, loss_fn, scaler)

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    save_checkpoint(checkpoint,filename=f'exp{exp_num}/{epoch}.pth.tar')

    # check accuracy
    val_fn(val_loader, model, device=dev,epoch=epoch)
    
    test_fn(test_loader,model,device=dev,epoch=epoch)

    # print some examples to a folder
    # save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=dev)




