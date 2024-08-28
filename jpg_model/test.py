import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet import ResNetCustom, BasicBlock
import os
from torchvision import models

def evaluate(model, val_loader, criterion, device):
    """

    :param model:
    :param val_loader:
    :param criterion:
    :return:
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    print('Finished Testng')
    return val_loss, val_acc


def is_folder_empty(folder):
    return len(os.listdir(folder)) == 0

def find_latest_model_file(folder):
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.pth')]
    if not files:
        return None
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return os.path.join(folder, latest_file)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename)['model_state_dict'])
    model.eval()

def single_folder_evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted.item())
    # print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_data_path = '../data/0827_test'
    model_path = "vertically_model_checkpoints"
    os.makedirs(model_path,exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小为224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        # transforms.RandomPerspective(),#随机透视变换
        # transforms.RandomAffine(degrees=(-85, 85)),#随机仿射变换
        transforms.RandomGrayscale(),  # 随机将图像转换为灰度图像
        # transforms.RandomRotation(degrees=(-85, 85)),#随机旋转图像一定角度
        transforms.ColorJitter(),  # 随机调整图像的亮度、对比度、饱和度和色调
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    test_dataset = datasets.ImageFolder(root= test_data_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # model = ResNetCustom(BasicBlock, [2, 2, 2, 2], num_classes=2)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    model = model.to(device)
    if is_folder_empty(model_path):
        print("model is empty")
    else:
        the_last_model = find_latest_model_file(model_path)
        load_model(model, the_last_model)
        print("loading model")
    criterion = nn.CrossEntropyLoss()
    # single_folder_evaluate(model, test_dataloader, criterion, device)
    evaluate(model, test_dataloader, criterion, device)