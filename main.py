#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

transform = transforms.Compose([ #Trandsform to preprocess the image
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class EmotionClassifier(nn.Module): #Model
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) #Convolution layer 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) #Convolution layer 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) #Convolution layer 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) #Convolution layer 
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1) #Convolution layer 
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1) #Convolution layer 
        self.fc1 = nn.Linear(1024 * 1 * 1, 1024) #Linear layer
        self.fc2 = nn.Linear(1024, 512) #Linear Layer
        self.fc3 = nn.Linear(512, 7)  # 7 output classes for 7 different emotions

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x)) #Activation function ReLu
        x = nn.functional.max_pool2d(x, 2)
        # print("maaxpool1")
        x = nn.functional.relu(self.conv2(x)) #Activation function ReLu
        x = nn.functional.max_pool2d(x, 2)
        # print("maaxpool2")
        x = nn.functional.relu(self.conv3(x)) #Activation function ReLu
        x = nn.functional.max_pool2d(x, 2)
        # print("maaxpool3")
        x = nn.functional.relu(self.conv4(x)) #Activation function ReLu
        x = nn.functional.max_pool2d(x, 2)
        # print("maaxpool4")
        x = nn.functional.relu(self.conv5(x)) #Activation function ReLu
        x = nn.functional.max_pool2d(x, 2)
        # print("maaxpool5")
        x = nn.functional.relu(self.conv6(x)) #Activation function ReLu
        x = torch.flatten(x,1) #Flattening the nuerons
        x = nn.functional.relu(self.fc1(x)) #Activation function ReLu
        x = nn.functional.relu(self.fc2(x)) #Activation function ReLu
        out = self.fc3(x)
        return out

if __name__ == '__main__':
    train_dataset = ImageFolder(r'C:\Users\varun\Downloads\Biometrics_finalProject\masked\train_masked', transform=transform)
    test_dataset = ImageFolder(r'C:\Users\varun\Downloads\Biometrics_finalProject\masked\test_masked', transform=transform)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    model = EmotionClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0.0
        model.train()

        for batch, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # print(images.shape)
            outputs = model(images)
            # print(predicted.dtype)
            # print(labels.dtype)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            # train_correct += (predicted == labels).sum().item()
        train_loss /= len(train_loader)
        # train_accuracy = 100.0 * train_correct / len(train_dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], 'f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_correct:.2f}%')

    script_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_path, 'model.pt')
    torch.save(model.state_dict(), model_path) 

    model.eval()
    test_correct = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = 100.0 * test_correct / len(test_dataset)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

