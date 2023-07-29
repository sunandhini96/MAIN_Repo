''' image transforms,
gradcam,
misclassification code '''
import math
from typing import NoReturn
from torchvision import datasets, transforms
import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

from models import resnet
# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def train_transform_function(mean,std):
    train_transform = A.Compose([A.Normalize(mean,std,always_apply=True),
                                 #A.PadIfNeeded(min_height=40,min_width=40,always_apply=True),
                                 A.RandomCrop(height=32,width=32,padding=4, always_apply=True),
                                 #A.HorizontalFlip(),
                                 #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,rotate_limit=15,p=0.5),
                                 A.CoarseDropout(max_holes=1,max_height=16,max_width=16, min_holes=1, min_height=16,min_width=16, fill_value=mean, mask_fill_value = None),
                                 ToTensorV2()

                                       ])
    return lambda img:train_transform(image=np.array(img))["image"]
# Test Phase transformations
def test_transform_function(mean,std):
      test_transform = A.Compose([
                                            #  transforms.Resize((28, 28)),
                                            #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),

                                            #transforms.ToTensor(),
                                            A.Normalize(mean,std),
                                            ToTensorV2()
                                            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ])
      return lambda img:test_transform(image=np.array(img))["image"]

def get_summary(model: 'object of model architecture', input_size: tuple) -> NoReturn:
    """
    Function to get the summary of the model architecture
    :param model: Object of model architecture class
    :param input_size: Input data shape (Channels, Height, Width)
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    summary(network, input_size=input_size)


def get_misclassified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()

    # List to store misclassified Images
    misclassified_data = []

    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:

            # Migrate the data to the device
            data, target = data.to(device), target.to(device)

            # Extract single image, label from the batch
            for image, label in zip(data, target):

                # Add batch dimension to the image
                image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(image)

                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)

                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data



