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
    return val_loss, val_acc
def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=500, save_interval=100):
    """

    :param model:
    :param train_loader:
    :param val_loader:
    :param optimizer:
    :param criterion:
    :param epochs:
    :param save_interval:
    :return:
    """
    # model.test()
    iteration = 0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iteration += 1

            # 输出loss和准确率
            if i % 10 == 9:  # 每间隔10次输出一次
                avg_loss = running_loss /8
                acc = correct / total
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')
                running_loss = 0.0
                correct = 0
                total = 0

                # 保存模型
            if iteration % save_interval == save_interval - 1:  # 每间隔save_interval次保存一次模型
                print("saving model.....")
                save_checkpoint(model, optimizer, epoch, folder="vertically_model_checkpoints", max_keep=3)
                # torch.save(model.state_dict(), f'model2_process/resnet_epoch_{epoch + 1}_step_{i + 1}.pth')
                # 在验证集上测试模型
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    print('Finished Training')
def is_folder_empty(folder):
    return len(os.listdir(folder)) == 0


def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()


def find_latest_model_file(folder):
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.pth')]
    if not files:
        return None
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return os.path.join(folder, latest_file)

def save_checkpoint(model, optimizer, epoch, folder="model_0826_checkpoints", max_keep=3):
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_path = os.path.join(folder, f"model_checkpoint_epoch_{epoch}.pth")
    # 保存模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)

    # 获取所有模型文件
    all_model_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pth')]
    # 根据创建时间排序，保留最新的`max_keep`个文件
    if len(all_model_files) > max_keep:
        all_model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        # 删除最老的文件
        for old_model in all_model_files[max_keep:]:
            os.remove(old_model)

def modify_resnet50_for_global_pooling(num_classes=5):
    # 加载预训练的ResNet50模型
    model = models.resnet50(pretrained=True)

    # 移除原有的全连接层
    model.fc = nn.Identity()  # 可以移除或者保留，具体看是否在后续使用中直接应用

    # 添加全局平均池化层，输出即为每个通道的平均值
    model.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    # 添加一个新的全连接层，用于分类九个类别
    num_features = 2048  # 使用原全连接层的输入特征数
    model.classifier = nn.Linear(num_features, num_classes)

    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data_path = "../data/0827"
    model_path = "model"
    os.makedirs(model_path,exist_ok=True)

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # 调整图片大小为224x224
        transforms.RandomHorizontalFlip(),#随机水平翻转
        transforms.RandomVerticalFlip(),#随机垂直翻转
        # transforms.RandomPerspective(),#随机透视变换
        # transforms.RandomAffine(degrees=(-85, 85)),#随机仿射变换
        transforms.RandomGrayscale(),#随机将图像转换为灰度图像
        # transforms.RandomRotation(degrees=(-85, 85)),#随机旋转图像一定角度
        transforms.ColorJitter(),#随机调整图像的亮度、对比度、饱和度和色调
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    train_dataset = datasets.ImageFolder(root= train_data_path, transform=transform)
    # test_dataset = datasets.ImageFolder(root='train_data', transform=transform)

    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # model = ResNetCustom(BasicBlock, [2, 2, 2, 2], num_classes=2)
    # model = models.resnet18(pretrained=True)
    model = models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
    #adamw
    optimizer = optim.AdamW(model.parameters(), lr=1e-04, weight_decay=0.0001)
    #adam
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #SGD
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 使用SGD优化器
    # 训练模型
    train(model, train_loader, val_loader, optimizer, criterion, device, epochs=5000, save_interval=100)

    #data (file://LUOHUI/data)