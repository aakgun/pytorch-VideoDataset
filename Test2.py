from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import ImageGrid
import os
from video_dataset import  VideoFrameDataset, ImglistToTensor


#########################################
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    videos_root = 'F:/git/pytorch-VideoDataset/demo_dataset'  # os.path.join(os.getcwd(), 'demo_dataset')  # Folder in which all videos lie in a specific structure
    annotation_file = 'F:/git/pytorch-VideoDataset/demo_dataset/annotations.txt'  # os.path.join(root, 'annotations.txt')  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)

    videos_root = os.path.join(os.getcwd(), 'demo_videos')
    annotation_file = os.path.join(videos_root, 'annotations.txt')

    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(299),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(299),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=5,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    for epoch in range(10):
        for video_batch, labels in dataloader:
            """
            Insert Training Code Here
            """
            print(labels)
            print("\nVideo Batch Tensor Size:", video_batch.size())
            print("Batch Labels Size:", labels.size())
            break
        break

#######################################################
#      Create dictionary for class indexes
#######################################################
classes = ["ApplyEyeMakeup","Archery"] #to store class values
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

params = {
#    "model": "resnet50",
    #"device": "cuda",
#    "lr": 0.001,
    "batch_size": 64,
    "num_workers": 4,
    "n_epochs": 10,
    "image_size": 256,
    "in_channels": 3,
    "num_classes": 5
}

image_paths = ["F:/git/pytorch-VideoDataset/demo_videos/ApplyEyeMakeup/frame000001.jpg",
               "F:/git/pytorch-VideoDataset/demo_videos/ApplyEyeMakeup/frame000002.jpg",
               "F:/git/pytorch-VideoDataset/demo_videos/ApplyEyeMakeup/frame000003.jpg",
                "F:/git/pytorch-VideoDataset/demo_videos/Archery/frame000001.jpg",
                "F:/git/pytorch-VideoDataset/demo_videos/Archery/frame000002.jpg",
                "F:/git/pytorch-VideoDataset/demo_videos/Archery/frame000003.jpg",
"F:/git/pytorch-VideoDataset/demo_videos/Archery/frame000004.jpg"
               ]

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=params["image_size"], width=params["image_size"]),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=params["image_size"], width=params["image_size"]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

from torch.utils.data import Dataset
class LandmarkDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('/')[-2]
        print(label)
        label = class_to_idx[label]
        print(image_filepath)
        if self.transform is not None:
            print("test:"+str(label))
            image = self.transform(image=image)["image"]

        return image, label

from torch.utils.data import DataLoader

train_dataset = LandmarkDataset(image_paths,train_transforms)
train_loader = DataLoader(
    train_dataset, batch_size=params["batch_size"], shuffle=True
)

train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True
)
next(iter(train_loader))[1].shape


train_dataset = LandmarkDataset(train_image_paths,train_transforms)
valid_dataset = LandmarkDataset(valid_image_paths,test_transforms) #test transforms are applied
test_dataset = LandmarkDataset(test_image_paths,test_transforms)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import torch
from models.C3D import C3D
from models.R2Plus1D import R2Plus1D
from models.R2Plus1D import R2Plus1D
NUM_CLASS = 2
model = C3D(NUM_CLASS)
model = R2Plus1D(NUM_CLASS, (2, 2, 2, 2))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, optimizer,torch.nn.CrossEntropyLoss(), train_loader,train_loader, epochs=5, device=device)


for epoch in range(10):
    i=0
    for video_batch, labels in train_loader:
        i = i + 1
        print("i:"+str(i))
        """
        Insert Training Code Here
        """
        print("labels" + str(labels))
        print("\nVideo Batch Tensor Size:", video_batch.size())
        print("Batch Labels Size:", labels.size())
        break
    break
