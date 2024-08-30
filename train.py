import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50
from torch.utils.data import DataLoader
import argparse


def parse_option():
    parser = argparse.ArgumentParser('Hyperparameter')

    parser.add_argument('--train_dir', type=str, default="./")
    parser.add_argument('--valid_dir', type=str, default="./")
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--save_dir', type=str, default='./save')

    # Optimizer Parameter
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)

    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)

    return opt


def test(model, dataloader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)

    print(
        f'Epoch {epoch}, Validation Loss: {total_loss:.4f}, Validation Acc: {total_acc:.4f}')

    return total_acc, total_loss


def train(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print(
        f'Epoch {epoch}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

    return epoch_acc, epoch_loss


def main():
    opt = parse_option()

    if opt.model == 'resnet18':
        model = resnet18(weights='DEFAULT')
    elif opt.model == 'resnet34':
        model = resnet34(weights='DEFAULT')
    elif opt.model == 'resnet50':
        model = resnet50(weights='DEFAULT')
    else:
        raise ValueError(f"Unsupported model: {opt.model}")

    model.fc = nn.Linear(model.fc.in_features, opt.class_num)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.learning_rate,
                                 weight_decay=opt.weight_decay, )

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root=opt.train_dir, train=True, download=True,
                                                      transform=transform)

    val_dataset = torchvision.datasets.FashionMNIST(root=opt.valid_dir, train=False, download=True,
                                                    transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    best_acc = 0.0

    for epoch in range(1, opt.epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch)
        val_acc, _ = test(model, val_loader, criterion, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(opt.save_dir, 'best_model.pth'))

    print("Finished !\n Best Accuracy : {}%".format(best_acc * 100))


if __name__ == '__main__':
    main()
