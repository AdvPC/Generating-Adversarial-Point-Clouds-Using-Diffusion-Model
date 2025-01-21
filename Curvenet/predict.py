import torch
import numpy as np
from Curvenet import CurveNet  # 确保正确导入CurveNet
from defense.drop_points import *

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = CurveNet(num_classes=40).to(device)  # num_classes根据实际情况调整
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('model.t7', map_location=device))
model.eval()

# 加载点云数据
point_cloud = np.load('../results/GEN_Ours_chair1_1711805889/out.npy')  # 确保文件路径正确
point_cloud = torch.tensor(point_cloud, dtype=torch.float).to(device)

srs =SRSDefense(drop_num=600)
point_cloud = srs(point_cloud)

# point_cloud = point_cloud.unsqueeze(0)  # 增加一个批次维度，因为模型期望的输入是(batch_size, N, 3)
point_cloud = point_cloud.permute(0, 2, 1)  # 调整为模型期望的形状(batch_size, 3, N)

# 进行预测
with torch.no_grad():
    logits = model(point_cloud)
    print(logits.shape)
    _, predicted_labels = torch.max(logits, 1)

# 打印分类结果
print("Predicted class:", predicted_labels.squeeze().cpu().numpy())



