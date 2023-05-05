import os
from model import VGG13
from data_setup import dataset_generator, create_dataloaders
from utils import train_step, test_step, save_model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T


# Set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Variable
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKER = os.cpu_count()

SRC_PATH = 'PetImages'
DEST_PATH = 'Dataset'
TRAIN_TEST_SIZE = [0.8, 0.2]
CREATE_DATASET = False  # Overwrite if already exist

CLASSES = sorted(os.listdir(SRC_PATH))

# Hyperparameter
SIZE = (224, 224)
LEARNING_RATE = 0.00001
BATCH_SIZE = 16
EPOCHS = 20

# Data transforms
train_tf = T.Compose([
    T.Resize(SIZE),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
test_tf = T.Compose([
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# Generate the Dataset
if not os.path.exists(DEST_PATH) or CREATE_DATASET:
    dataset_generator(src_path=SRC_PATH,
                      dest_path=DEST_PATH,
                      split_size=TRAIN_TEST_SIZE,
                      image_format=['jpg'])

# Create the DataLoader
train_loader, test_loader = create_dataloaders(data_path=DEST_PATH,
                                               batch_size=BATCH_SIZE,
                                               train_transform=train_tf,
                                               test_transform=test_tf,
                                               num_worker=NUM_WORKER)

# Model infor
model = VGG13().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
MODEL_SAVE_STEP = 2

# Training loop
for epoch in range(EPOCHS):
    train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, DEVICE)

    test_loss, test_acc = test_step(model, test_loader, criterion, DEVICE)

    scheduler.step(test_loss)

    print(f"Epoch: {epoch+1} | Lr: {optimizer.param_groups[0]['lr']}                              ")
    print(f"  |- loss: {train_loss:.4f}  acc: {train_acc:.4f} | test_loss: {test_loss:.4f}  test_acc: {test_acc:.4f}")

    if (epoch+1) % MODEL_SAVE_STEP == 0:
        save_model(model=model, target_dir="Models", model_name=f"E{epoch+1}_L{test_loss:.4f}_A{test_acc:.2f}.pth")
