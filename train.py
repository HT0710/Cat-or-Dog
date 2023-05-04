import os
from model import TinyVGG
from data_setup import dataset_generator, create_dataloaders
from utils import train_step, test_step, save_model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T

torch.manual_seed(42)
torch.cuda.manual_seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DGC = DATASET_GENERATE_CONFIG = {
    "src_path": 'PetImages',
    "dest_path": 'Dataset',
    "val_size": 0.2,
    "test_size": 0,
    "replace": False,
    "random_seed": 42
}
CLASSES = sorted(os.listdir(DGC["src_path"]))
NUM_WORKER = os.cpu_count()
SIZE = (128, 128)
BATCH_SIZE = 50
EPOCHS = 0
LEARNING_RATE = 0.001

transforms = T.Compose([
    T.Resize(SIZE),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    T.ToTensor()
])    


if not os.path.exists(DGC["dest_path"]) or DGC["replace"]:
    dataset_generator(DATASET_GENERATE_CONFIG)

train_dataloader, val_dataloader, test_dataloader = create_dataloaders(data_path=DGC['dest_path'], 
                                                                       batch_size=BATCH_SIZE, 
                                                                       train_transform=transforms,
                                                                       num_worker=NUM_WORKER)

model = TinyVGG().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

for epoch in range(EPOCHS):
    train_loss, train_acc = train_step(model, train_dataloader, criterion, optimizer, DEVICE)

    val_loss, val_acc = test_step(model, val_dataloader, criterion, DEVICE)

    scheduler.step()

    print(f"Epoch: {epoch+1} | Lr: {optimizer.param_groups[0]['lr']}")
    print(f"  |- loss: {train_loss:.4f} acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

    if (epoch+1) % 5 == 0:
        test_loss, test_acc = test_step(model, test_dataloader, criterion, DEVICE)
        save_model(model=model, target_dir="saved", model_name=f"{epoch+1}_epoch_{test_loss:.4f}_{test_acc:.2f}.pth")
