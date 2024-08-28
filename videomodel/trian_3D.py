from model_3D import ResNet3D
import torch
import torch.nn as nn
from dataload import get_data
import torch.optim as optim
import os


class CustomLoss(nn.Module):
    def __init__(self, a, delta, b, K, tau, rho):
        super(CustomLoss, self).__init__()
        self.a = a
        self.delta = delta
        self.b = b
        self.K = K
        self.tau = tau
        self.rho = rho

    def forward(self, y_pred, y_true):
        # 计算 e_i
        e_i = torch.abs((y_true - y_pred) / y_pred)

        # 计算 W(y_i)
        condition1 = (y_true < self.a - self.delta) | (y_true > self.b + self.delta)
        condition2 = (y_true >= self.a - self.delta) & (y_true < self.a)
        condition3 = (y_true >= self.a) & (y_true < self.b)
        condition4 = (y_true >= self.b) & (y_true <= self.b + self.delta)

        W = torch.where(condition1, torch.tensor(1.0), torch.tensor(1.0 + self.K * self.delta))
        W = torch.where(condition2, torch.tensor(1.0 + self.K * (y_true - self.a + self.delta)), W)
        W = torch.where(condition4, torch.tensor(1.0 + self.K * self.delta - self.K * (y_true - self.b)), W)

        # 计算 L_i
        L = torch.where(e_i < self.tau, e_i, e_i ** self.rho)

        # 计算总损失
        numerator = torch.sum(W * torch.log(1 + self.K * e_i) * L)
        denominator = torch.sum(W * torch.log(1 + self.K * e_i))
        loss = numerator / denominator

        return loss


def train_model(model, train_loader, criterion, optimizer, start_epoch, num_epochs=100000,N=100):
    for epoch in range(start_epoch, num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            if inputs.size(0) == 0 or labels.size(0) == 0:
                print(f"Skipping empty batch {i}")
                continue
            optimizer.zero_grad()
            inputs = inputs.permute(0, 2, 1, 3, 4).cuda()
            model = model.cuda()
            outputs = model(inputs)
            # print(type(labels))
            labels = labels.float().cuda()
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {loss.item()}')
        if epoch % N == N-1 :
            save_checkpoint(model, optimizer, epoch)
        save_checkpoint(model, optimizer, epoch)
# def predict(model, inputs):
#     model.eval()
#     with torch.no_grad():
#         outputs = model(inputs)
#         return outputs


def save_checkpoint(model, optimizer, epoch, folder="model_checkpoints", max_keep=3):
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

    print(f"Model saved to {model_path}, old models cleared.")


# def load_latest_checkpoint(folder="model_checkpoints", model=None, optimizer=None):
#     checkpoint_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pth')]
#     if not checkpoint_files:
#         print("No checkpoints found.")
#         return None
#
#     latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
#     checkpoint = torch.load(latest_checkpoint)
#
#     if model is not None and 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     if optimizer is not None and 'optimizer_state_dict' in checkpoint:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#     print(f"Loaded checkpoint '{latest_checkpoint}' saved at epoch {checkpoint['epoch']}.")
#     return checkpoint


def predict_with_latest_model(input_data, folder="model_checkpoints", model=None):
    if model is None:
        model = ResNet3D()  # 重新实例化你的模型

    # 寻找并加载最新的模型
    latest_checkpoint = max((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pth')),
                            key=os.path.getmtime)
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        prediction = model(input_data)

    print(f"Loaded model from {latest_checkpoint} for prediction.")
    return prediction


def load_latest_checkpoint(checkpoint_dir, model, optimizer):
    # 获取目录中所有文件
    all_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not all_checkpoints:
        raise FileNotFoundError("No checkpoint found.")
    # 找到最新的文件
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    # 加载检查点
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint '{latest_checkpoint}' (epoch {epoch})")
    return model, optimizer, epoch
if __name__=='__main__':
    video_path = "../../train_video"
    dataloader = get_data(video_path)
    model = ResNet3D()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 加载最新的模型
    model_path = " "
    try:
        model, optimizer, start_epoch = load_latest_checkpoint(model_path, model, optimizer)
    except FileNotFoundError:
        start_epoch = 0  # 如果没有找到检查点，从头开始训练
    # dataloader = get_data(video_files,labels)
    # Assuming `dataloader` is defined
    train_model(model, dataloader, criterion, optimizer, start_epoch)