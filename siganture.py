import torch
import gradio as gr


import torchvision.transforms as transforms 
from PIL import Image
import numpy as np 
import pandas as pd
import os

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import random
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from main import parse_cheque_with_donut, load_image, load_donut_model_and_processor, prepare_data_using_processor, spell_check, match_legal_and_courstesy_amount



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()


        self.conv1=nn.Conv2d(1,50,kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
         # L1 ImgIn shape=(?, 28, 28, 1)      # (n-f+2*p/s)+1
        #    Conv     -> (?, 24, 24, 50)
        #    Pool     -> (?, 12, 12, 50)


        self.conv2 = nn.Conv2d(50,60, kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        # L2 ImgIn shape=(?, 12, 12, 50)
        #    Conv      ->(?, 8, 8, 60)
        #    Pool      ->(?, 4, 4, 60)


        self.conv3 = nn.Conv2d(60, 80,  kernel_size = 3)
        # L3 ImgIn shape=(?, 4, 4, 60)
        #    Conv      ->(?, 2, 2, 80)



        self.batch_norm1 = nn.BatchNorm2d(50)
        self.batch_norm2 = nn.BatchNorm2d(60)

#         self.dropout1 = nn.Dropout2d()

        # L4 FC 2*2*80 inputs -> 250 outputs
        self.fc1 = nn.Linear(32000, 128)
        self.fc2 = nn.Linear(128, 2)




    def forward1(self,x):
        x=self.conv1(x)
        x = self.batch_norm1(x)
        x=F.relu(x)
        x=self.pool1(x)

        x=self.conv2(x)
        x = self.batch_norm2(x)
        x=F.relu(x)
        x=self.pool2(x)

        x=self.conv3(x)
        x=F.relu(x)
#         print(x.size())
        x = x.view(x.size()[0], -1)
#         print('Output2')
#         print(x.size()) #32000 thats why the input of fully connected layer is 32000
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward1(input1)
        # forward pass of input 2
        output2 = self.forward1(input2)

        return output1, output2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = SiameseNetwork().to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))

def preprocess_image(image_path, crop_box=None):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Grayscale(),  
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    if crop_box:
        image = image.crop(crop_box)  
    image = transform(image)
    return image

# Function to test with two signature images
def test_with_images(image_path1,crop_box1, image_path2):
    # Preprocess the images
    image1 = preprocess_image(image_path1,crop_box1).to(device)
    image2 = preprocess_image(image_path2).to(device)

    # Pass the images through the model
    with torch.no_grad():
        output1, output2 = model(image1.unsqueeze(0), image2.unsqueeze(0))

    # Compute the Euclidean distance
    euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2).item()

    # Predict the label based on the distance
    threshold = 0.4  # Adjust the threshold as needed
    predicted_label = "Original Pair Of Signature" if euclidean_distance < threshold else "Forged Pair Of Signature"

    return euclidean_distance, predicted_label



def analyze_cheque(image_path):
    
    
    # Perform cheque analysis
    image_path1 = image_path
    crop_box1 = (338, 142, 491, 204)  # You may adjust this crop box as needed
    folder_path = "C:\\Users\\manso\\OneDrive\\Bureau\\Donut\\paye_names\\"

    # Parse cheque with Donut
    payee_name, amt_in_words, amt_in_figures, bank_name, cheque_date, matching_amts, stale_cheque = parse_cheque_with_donut(image_path1)

    # Find matching image
    matching_image = None
    for file_name in os.listdir(folder_path):
        if payee_name in file_name:
            matching_image = folder_path + file_name
            break

    if matching_image:
        image_path2 = matching_image
        euclidean_distance, predicted_label = test_with_images(image_path1, crop_box1, image_path2)
    else:
        euclidean_distance, predicted_label = None, None

    return (
            f"Payee Name: {payee_name}", 
            f"Amount in Words: {amt_in_words}", 
            f"Amount in Figures: {amt_in_figures}", 
            f"Bank Name: {bank_name}", 
            f"Cheque Date: {cheque_date}", 
            f"Matching Amounts: {matching_amts}", 
            f"Stale Cheque: {stale_cheque}", 
            f"Euclidean Distance: {euclidean_distance}", 
            f"Predicted Label: {predicted_label}"
        )


# Create Gradio interface
interface = gr.Interface(
    analyze_cheque, 
    inputs=gr.inputs.Image(type="filepath", label="Upload Image"), 
    outputs=["text", "text", "text", "text", "text", "text", "text", "text", "text"], 
    
    title="Cheque Verification",
    description="Upload an image of the cheque to analyze its details"
)
interface.launch(share=True)


