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
from albumentations.pytorch.transforms import ToTensorV2

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
                                 A.RandomCrop(height=32,width=32, always_apply=True),
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

# -------------------- GradCam --------------------
def display_gradcam_output(data: list,
                           classes: list[str],
                           inv_normalize: transforms.Normalize,
                           model: 'DL Model',
                           target_layers: list['model_layer'],
                           targets=None,
                           number_of_samples: int = 10,
                           transparency: float = 0.60):
    """
    Function to visualize GradCam output on the data
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param model: Model architecture
    :param target_layers: Layers on which GradCam should be executed
    :param targets: Classes to be focused on for GradCam
    :param number_of_samples: Number of images to print
    :param transparency: Weight of Normal image when mixed with activations
    """
    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    # Create an object for GradCam
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # Iterate over number of specified images
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Get back the original image
        img = input_tensor.squeeze(0).to('cpu')
        img = inv_normalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()

        # Mix the activations on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)

        # Display the images on the plot
        plt.imshow(visualization)
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])

def get_classified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()

    # List to store misclassified Images
    classified_data = []

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
                if pred == label:
                    classified_data.append((image, label, pred))
    return classified_data

def display_cifar_misclassified_data(misclassified_data, classes, inv_normalize, number_of_samples=20):
    """
    Display misclassified CIFAR-10 data with predicted and true labels.

    Parameters:
        misclassified_data (list): A list of tuples, each containing (image, predicted_label, true_label).
        classes (list or dict): A list or dictionary containing the class names.
        inv_normalize (torchvision.transforms.Normalize): The inverse normalization transformation to convert images back to their original scale.
        number_of_samples (int): The number of misclassified samples to display (default is 20).
    """

    num_samples = min(number_of_samples, len(misclassified_data))

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.5)

    for i in range(num_samples):
        ax = axes[i // 5, i % 5]
        data_tuple = misclassified_data[i]

        if len(data_tuple) == 3:
            image, predicted_label, true_label = data_tuple

            # Convert the image from a PyTorch tensor to a numpy array and apply inverse normalization
            image = inv_normalize(torch.unsqueeze(image, 0))  # Add a batch dimension before normalization
            image = image.squeeze().cpu().numpy().transpose(1, 2, 0)  # Remove batch dimension after normalization
            image = np.clip(image, 0, 1)  # Clip values to ensure they are in the valid range [0, 1]

            # Display the image
            ax.imshow(image)
            ax.set_title(f'Pred: {classes[predicted_label]}\nTrue: {classes[true_label]}')
            ax.axis('off')
        else:
            print(f"Data at index {i} is not in the expected tuple format.")

    plt.show()

# Assuming the `get_misclassified_data` function returns a list of tuples, each containing (image, predicted_label, true_label).
# Also, `classes` should be a list of class names (e.g., ['cat', 'dog', ...]) or a dictionary with class indices as keys and class names as values.

# Example usage:
# display_cifar_misclassified_data(misclassified_data, classes, inv_normalize, number_of_samples=20)


    



