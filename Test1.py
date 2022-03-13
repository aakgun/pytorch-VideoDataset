import torch
import torch.nn as nn
from datasets import VideoDataset,VideoLabelDataset

import transforms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
#from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True

dataset = VideoDataset(
        "data/example_video_file3.csv",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/trainlist01.txt",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/testlist01.txt",
        transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
        )

dataset = VideoDataset(
    "data/example_video_file3.csv",
    transform=torchvision.transforms.Compose([
        transforms.VideoFilePathToTensor(max_len=50, padding_mode='last'),
        transforms.VideoRandomCrop([512, 512]),
        transforms.VideoResize([256, 256]),
    ])
)

dataset = VideoLabelDataset(
        "data/example_video_file_with_label21.csv",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/trainlist01.txt",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/testlist01.txt",
        transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
        )

dataset = VideoLabelDataset(
        "data/example_video_file_with_label21.csv",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/trainlist01.txt",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/testlist01.txt",
        transform=torchvision.transforms.Compose([
                transforms.VideoFilePathToTensor(),
                #transforms.VideoRandomCrop([512, 512]),
                transforms.VideoResize([64, 64]),
            ])
        )

dataset_train = VideoLabelDataset(
        "data/example_video_file_with_label2.csv",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/trainlist01.txt",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/testlist01.txt",
        transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
        )

dataset_test = VideoLabelDataset(
        "data/example_video_file_with_label2.csv",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/trainlist01.txt",
        #"C:/git/video-clip-order-prediction/data/ucf101/split/testlist01.txt",
        transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
        )

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
for video,label in data_loader:
    print(video.size())
    print(label)

for videos in data_loader:
    print(videos.size())

from models.C3D import C3D
from models.R2Plus1D import R2Plus1D
from models.R2Plus1D import R2Plus1D
NUM_CLASS = 2
model = C3D(NUM_CLASS)
model = R2Plus1D(NUM_CLASS, (2, 2, 2, 2))
model.to(device)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
simplenet = SimpleNet()
simplenet.to(device)

model = simplenet


for batch in data_loader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    output = model(inputs)

#################################
    epochs=10
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = torch.nn.CrossEntropyLoss(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(data_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = torch.nn.CrossEntropyLoss(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(data_loader.dataset)

        print(
            'Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
                                                                                                  valid_loss,
                                                                                                  num_correct / num_examples))

#################################################

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=10, device="gpu"):
    print("Train")
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            print("epoch:"+str(epoch))
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print(
            'Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
                                                                                                  valid_loss,
                                                                                                  num_correct / num_examples))
print("test")

optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, optimizer,torch.nn.CrossEntropyLoss(), data_loader,data_loader, epochs=5, device=device)

import torch
from datasets import VideoDataset
import transforms

csv_file = "data/example_video_file.csv"
import pandas as pd

dataframe = pd.read_csv(csv_file)
for x in dataframe.index:
    print(dataframe.iloc[x].path)

for x in dataframe.iterrows():
    print(x[1])

video = dataframe.iloc[index].path
video = self.transform(video)

dataset = VideoDataset(
        "./data/example_video_folder2.csv",
        transform=transforms.VideoFolderPathToTensor()  # See more options at transforms.py
)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
for videos in data_loader:
    print(videos.size())