# Assignment2

## Prerequisites
+ Python 3.8 (Anaconda)
+ PyTorch 1.8.1
+ CUDA 10.2

## Environmental Setup
```
$ conda create --name ENV_NAME python=3.8
$ conda activate ENV_NAME
$ conda install pytorch==1.8.1 torchvision cudatoolkit=10.2 -c pytorch
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Data Preparation
Get the X-ray Tuberculosis dataset from [Kaggle](https://www.kaggle.com/datasets/usmanshams/tbx-11)

## X-ray classification ResNet50 
HC701 Assignment2 Task2.2  
**Dataset**: X-ray Tuberculosis dataset  
**Task**: There are 3800 healthy x-ray
images and 800 TB x-rays. Use ResNet50 to do binary classification.
1. Split data for training and testing. We use the first (sorted in an ascending order by ID) 20% of images per class for testing and the remaining 80% for training, storing in 'task21.csv' CSV files and report the number and the range of filenames for each class in training and testing sets.

2. 5 Experiments: We implement 5 experiments with different data augmentation strategies and model structure to improve classificator performance. We report the best accuracy, f1 score, confusion matrix as well as the number of parameter, FLOPs in each experiment. 

## Data augmentation strategies and neural network architecture
For neural network architecture, we select **ResNet50**.    

ResNet50 is a convolutional neural network architecture that has demonstrated state-of-the-art performance in image classification tasks. The main benefit of ResNet50 is its ability to train very deep neural networks with hundreds of layers without encountering the vanishing gradient problem, which is a common issue that arises when training very deep neural networks.  

ResNet50 achieves this by introducing residual connections, which allow the network to bypass certain layers and make direct connections between input and output layers. This architecture enables the network to learn more complex features and make better predictions, leading to improved accuracy in image classification tasks.  

For data augmentation strategies, we choose **ColorJitter**, **RandomHorizontalFlip** and **RandomVerticalFlip**

**ColorJitter(brightness=0,contrast=0,saturation=0,hue=0)** is a type of transformation that changes the color of an image by applying random perturbations to its hue, saturation, brightness, and contrast. This technique is used to create additional training data that can help improve the accuracy and robustness of machine learning models.

**RandomHorizontalFlip(p=0.5)/RandomVerticalFlip(p=0.5)**  involves randomly flipping images horizontally/vertically along the x-axis/y-axis. This technique can be applied to image datasets during training to increase the size of the dataset and improve the model's ability to generalize to new, unseen data. By randomly flipping images horizontally/vertically, the model learns to recognize the same object or pattern, whether it appears upside down or right-side up.

## Experiments Setting
run task2.py to do five experiments. 
```
if exp_num == 1:
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    model = OFAResNets(n_classes=2,dropout_rate=0.5, depth_list=[0, 1, 2],expand_ratio_list=[0.2, 0.25, 0.35],width_mult_list=[0.65, 0.8, 1.0])
elif exp_num==2:
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    model = OFAResNets(n_classes=2,dropout_rate=0.5, depth_list=[0, 1, 2],expand_ratio_list=[0.2, 0.25, 0.35],width_mult_list=[0.65, 0.8, 1.0])
elif exp_num == 3:
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    model = OFAResNets(n_classes=2,dropout_rate=0.5, depth_list=[0, 1, 2],expand_ratio_list=[0.2, 0.25, 0.35],width_mult_list=[0.65, 0.8, 1.0])
elif exp_num == 4:
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    model = OFAResNets(n_classes=2,dropout_rate=0.5, depth_list=[0, 1], expand_ratio_list=[0.2, 0.25], width_mult_list=[0.65, 0.8])
elif exp_num ==5:
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    model = OFAResNets(n_classes=2,dropout_rate=0.5, depth_list=[0, 1], expand_ratio_list=[0.2, 0.3], width_mult_list=[0.8, 0.8])
```

## Results
|Exp|Parameter|Flops|Accuracy |F1 score| Confusion Matrix|
|--|--|--|--|--|--|
|1|46061090|235.205M|91.63%|0.9033|[[752 8],[79 81]]|
|2|46061090|235.205M|90.22%|0.8878|[[756 4],[86 74]]|
|3|46061090|235.205M|92.50%|0.9167|[[759 1],[68 92]]|
|4|14228282|142.557M|73.26%|0.7271|[[645 115],[131 29]]|
|5|18469978|148.892M|96.96%|0.9691|[[752 8],[20 140]]|

