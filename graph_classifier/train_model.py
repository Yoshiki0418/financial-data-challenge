import os
import numpy as np
import hydra
from omegaconf import DictConfig
import torch.nn.functional as F
from termcolor import cprint
import torch
from tqdm import tqdm
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.utils import set_seed
from src.dataset import Image_Dataset
from src.dataset import get_image_paths_and_labels
from src.model import ResNet50BinaryClassifier


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    #--------------------------------
    #          Dataloader
    #--------------------------------
    loader_args = {"batch_size": args.model.batch_size, "num_workers": args.num_workers}

    image_paths, labels = get_image_paths_and_labels("data")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor()     
    ])

    train_set = Image_Dataset("train", train_paths, train_labels, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, **loader_args, shuffle=True)

    val_set = Image_Dataset("val", val_paths, val_labels, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, **loader_args, shuffle=False)

    #--------------------------------
    #            Model
    #--------------------------------
    model = ResNet50BinaryClassifier().to(args.device)

    #--------------------------------
    #          Optimizer
    #--------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.model.lr)

    #--------------------------------
    #     Start training
    #--------------------------------
    accuracy = Accuracy(task="binary").to(args.device)

    max_val_acc = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(args.model.epochs):
        print(f"Epoch {epoch+1}/{args.model.epochs}")

        train_loss, val_loss = [], []
        
        # Accuracyのリセット
        accuracy.reset()

        model.train()
        for X, y in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X).view(-1)  # (batch_size, 1) → (batch_size)
            
            loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracyの更新
            accuracy.update((y_pred > 0).float(), y)

        # 最終的なTrain Accuracyの計算
        train_epoch_acc = accuracy.compute().item()

        model.eval()
        accuracy.reset()
        y_true_list, y_pred_list = [], []
        for X, y in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            with torch.no_grad():
                y_pred = model(X).view(-1)

            v_loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
            val_loss.append(v_loss.item())

            accuracy.update((y_pred > 0).float(), y)

            y_true_list.append(y.cpu().numpy()) 
            y_pred_list.append((y_pred > 0).cpu().numpy())  

        # 最終的なValidation Accuracyの計算
        val_epoch_acc = accuracy.compute().item()

        y_true_list = np.concatenate(y_true_list).ravel()
        y_pred_list = np.concatenate(y_pred_list).ravel()

        print(f"Epoch {epoch+1}/{args.model.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {train_epoch_acc:.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {val_epoch_acc:.3f}")

        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        
        if val_epoch_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = val_epoch_acc

        train_loss_history.append(np.mean(train_loss))
        val_loss_history.append(np.mean(val_loss))

if __name__ == "__main__":
    run()
