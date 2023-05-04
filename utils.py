import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path



def train_step(
        model: nn.Module, 
        dataloader: DataLoader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        device: torch.device
):
    """Train function"""

    model.train()

    train_loss, train_acc = 0, 0
    data_size = len(dataloader)

    for step, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X).squeeze(1)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_acc += preds.eq(y).sum().item()/len(outputs)

    
    train_loss = train_loss / data_size
    train_acc = train_acc / data_size

    return train_loss, train_acc


def test_step(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device       
):
    """Test function"""

    model.eval()

    test_loss, test_acc = 0, 0
    data_size = len(dataloader)

    with torch.inference_mode():
        for step, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze(1)
            loss = criterion(outputs, y)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_acc += preds.eq(y).sum().item()/len(outputs)

    test_loss = test_loss / data_size
    test_acc = test_acc / data_size

    return test_loss, test_acc


def save_model(model: nn.Module, target_dir: str, model_name: str):

  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  torch.save(obj=model.state_dict(), f=model_save_path)
