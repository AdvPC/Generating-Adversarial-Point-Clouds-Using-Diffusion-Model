import torch
import torch.nn.functional as F
from torchvision import transforms
from model import PointNet  # 假设PointNet模型在model.py中定义


# 定义Chamfer距离计算函数
def chamfer_distance(x, y):
    # 计算x到y的距离矩阵
    dist_x_y = torch.cdist(x, y, p=2)  # 使用torch.cdist计算欧氏距离矩阵

    # 计算y到x的距离矩阵
    dist_y_x = torch.cdist(y, x, p=2)

    # 计算x到y的最小距离
    min_dist_x_y = torch.min(dist_x_y, dim=1)[0]

    # 计算y到x的最小距离
    min_dist_y_x = torch.min(dist_y_x, dim=1)[0]

    # 返回平均最小距离
    return torch.mean(min_dist_x_y) + torch.mean(min_dist_y_x)


# 加载PointNet模型
model = PointNet()
model.load_state_dict(torch.load('pointnet_model.pth'))
model.eval()

# 加载测试数据集
test_dataset = ModelNet40Dataset(transform=transforms.ToTensor())  # 假设数据集类为ModelNet40Dataset，并且有适当的转换
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 选择一个测试样本
sample, target = next(iter(test_loader))


# 定义IFGSM攻击函数
def ifgsm_attack(model, data, target, epsilon, alpha, num_iter):
    # 设置为评估模式
    model.eval()

    # 对输入数据进行梯度追踪
    data.requires_grad = True

    for i in range(num_iter):
        # 正向传播
        output = model(data)

        # 计算损失
        loss = nn.CrossEntropyLoss()(output, target)

        # 反向传播并获取输入数据的梯度
        model.zero_grad()
        loss.backward()

        # 对输入数据应用扰动
        with torch.no_grad():
            # 使用输入数据的梯度方向进行扰动
            perturbation = alpha * torch.sign(data.grad)
            data = torch.clamp(data + perturbation, 0, 1)

            # 将扰动限制在epsilon范围内
            data = torch.clamp(data, 0, 1)

    return data


# 设置epsilon、alpha和迭代次数
epsilon = 0.03
alpha = 0.01
num_iter = 10

# 使用IFGSM攻击生成对抗样本
adversarial_sample = ifgsm_attack(model, sample, target, epsilon, alpha, num_iter)

# 计算Chamfer距离
cd = chamfer_distance(sample.squeeze(0), adversarial_sample.squeeze(0))
print("Chamfer Distance:", cd.item())
