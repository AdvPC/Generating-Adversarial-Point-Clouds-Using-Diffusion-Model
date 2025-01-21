import argparse
import math

from scipy.spatial.distance import cdist
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler

# from evaluation import EMD_CD
from pointnet2_cls_ssg import *
from models.encoders import PointNetEncoder
from models.vae_flow import FlowVAE
from pointnet2_cls_ssg import *
from utils.data import get_data_iterator
from utils.dataset import *
from utils.misc import *
from utils_v2.model_utils import *
from evaluation.set_distance import ChamferDistance, HausdorffDistance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
def chamfer_distance(pointcloud1, pointcloud2):
        """
        计算两个点云之间的Chamfer距离
        输入:
            pointcloud1: 第一个点云, [B, N, 3]
            pointcloud2: 第二个点云, [B, M, 3]
        返回:
            chamfer_loss: 两个点云之间的Chamfer距离
        """
        B, N, _ = pointcloud1.shape
        _, M, _ = pointcloud2.shape

        # 计算每个点到另一个点云中所有点的距离
        dist1 = torch.cdist(pointcloud1, pointcloud2, p=2)  # [B, N, M]
        dist2 = torch.cdist(pointcloud2, pointcloud1, p=2)  # [B, M, N]

        # 取最小距离
        min_dist1, _ = torch.min(dist1, dim=2)  # [B, N]
        min_dist2, _ = torch.min(dist2, dim=2)  # [B, M]

        # 计算Chamfer距离
        chamfer_loss = torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)  # [B]

        # 取平均
        chamfer_loss = torch.mean(chamfer_loss)  # Scalar

        return chamfer_loss


def calculate_mse(file1_path, file2_path):
    # Load point clouds from the files
    point_cloud1 = np.load(file1_path)
    point_cloud2 = np.load(file2_path)

    # Check if point clouds have the same shape
    if point_cloud1.shape != point_cloud2.shape:
        raise ValueError("Point clouds have different shapes.")

    # Compute mean squared error (MSE)
    mse = np.mean(np.square(point_cloud1 - point_cloud2))

    return mse
# 创建Chamfer距离和Hausdorff距离的实例
chamfer_dist = ChamferDistance()
hausdorff_dist = HausdorffDistance()

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point




# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/GEN_chair.pt')
parser.add_argument('--categories', type=str_list, default=['desk1'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--latent_dim', type=int, default=256)
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/modelnet.h5')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--scale_mode', type=str, default='shape_unit')
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=500)
args = parser.parse_args()

if __name__ == '__main__':
    # Logging
    save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger('test', save_dir)

    # Datasets and loaders
    test_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=args.normalize,
    )
    # Assuming `test_dset` is your dataset
    num_samples = 1
    indices = np.random.choice(len(test_dset), num_samples, replace=False)
    sampler = SubsetRandomSampler(indices)
    test_loader = DataLoader(test_dset, sampler=sampler, batch_size=args.batch_size,drop_last=True, num_workers=0)

    # Checkpoint
    ckpt = torch.load(args.ckpt)
    model = FlowVAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])

    modelp = get_model(num_class=40, normal_channel=False).to(args.device)
    checkpoint = torch.load('best_model.pth')
    modelp.load_state_dict(checkpoint['model_state_dict'])
    modelp.eval()


    # Reference Point Clouds
    gen_pcs = []
    ref_pcs = []
    cd_losses = []
    hd_losses = []
    for i, data in enumerate(tqdm(test_loader)):
        ref = data['pointcloud'].to(args.device)
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        x = model.sample1(ref, z, flexibility=ckpt['args'].flexibility)

        # 确保 ref 和 x 不需要梯度
        ref = ref.detach().clone()
        x = x.detach().clone()

        # 初始化噪声
        noise = x - ref
        noise = noise.requires_grad_(True)

        dcd = calc_dcd(ref, x)

        mse = F.mse_loss(ref,x)

        # 执行预测，并计算ref的梯度
        point_cloud = ref.transpose(2, 1)
        prediction, _ = modelp(point_cloud)

        # 获取 ref 的预测标签最大值
        prediction_max = prediction.max(dim=1)[0].mean()  # 获取最大值

        optimizer = torch.optim.Adam([noise], lr=0.01)  # 创建一个优化器来更新 noise

        for iteration in range(999):  # 增加迭代次数
            optimizer.zero_grad()  # 清空梯度

            # 计算预测的梯度方向，并进行反向传播
            prediction_max.backward(retain_graph=True)  # 这里生成梯度

            # 执行预测，并计算ref的梯度
            # point_x = x.transpose(2, 1)
            # pred_x, _ = modelp(point_x)
            #
            # mse = F.mse_loss(prediction,pred_x)

            # 计算 loss，确保噪音参与损失计算
            total_loss = 4 * dcd + 0.5 * mse + 0.1 * torch.mean(noise ** 2)

            # 使用 total_loss 进行反向传播，确保生成噪音的梯度
            total_loss.backward()

            # 更新 noise：确保噪音方向与 ref 的梯度方向相反
            with torch.no_grad():
                noise -= 0.01 * torch.sign(noise.grad)  # 使用梯度方向反转更新噪音

            # 将噪音限制在 [-1, 1] 范围内
            noise.data = torch.clamp(noise.data, -0.32, 0.32)

            # 生成新的对抗样本 x
            x = x + noise

            # 重新计算生成对抗样本的损失
            dcd = calc_dcd(ref, x)
            point_cloud1 = x.transpose(2, 1)
            prediction1, _ = modelp(point_cloud1)

            # 确保 prediction1 和 prediction 的最大值不同
            if prediction1.max(dim=1)[1] != prediction.max(dim=1)[1]:
                print("攻击成功", prediction1.max(dim=1)[1], prediction.max(dim=1)[1])
                break  # 如果满足条件，跳出循环
            else:
                print("攻击失败")

        # 输出最终结果
        print(f"最终的 total_loss: {total_loss.item()}")

        gen_pcs.append(x.detach().cpu())
        ref_pcs.append(ref.detach().cpu())

    gen_pcs = torch.cat(gen_pcs,dim=0)
    ref_pcs = torch.cat(ref_pcs,dim=0)


    np.save(os.path.join(save_dir, 'out.npy'), gen_pcs.numpy())
    np.save(os.path.join(save_dir, 'ref.npy'), ref_pcs.numpy())
# 计算点云之间的距离
cd_losses = []
hd_losses = []
for i in tqdm(range(len(gen_pcs)), desc="Calculating distances"):
    # 取出1024个点
    gen_pc = gen_pcs[i].unsqueeze(0)  # 添加一个维度，变为 [1, N1, 3]
    ref_pc = ref_pcs[i].unsqueeze(0)  # 添加一个维度，变为 [1, N2, 3]

    # 计算 Chamfer 距离
    cd_loss = chamfer_dist(gen_pc.to(args.device), ref_pc.to(args.device))
    cd_losses.append(cd_loss.item())


    # 计算 Hausdorff 距离
    hd_loss = hausdorff_dist(gen_pc.to(args.device), ref_pc.to(args.device))
    hd_losses.append(hd_loss.item())

# 计算距离的平均值
average_cd = np.mean(cd_losses)
average_hd = np.mean(hd_losses)
avd_cd = np.mean(cd_losses)
avd_hd = np.mean(hd_losses)
mse = calculate_mse(os.path.join(save_dir, 'out.npy'), os.path.join(save_dir, 'ref.npy'))
cdc = calc_dcd(gen_pcs.to(args.device),ref_pcs.to(args.device))
logger.info('MSE: %.12f' % mse)
logger.info('CDC: %.12f' % cdc)
logger.info('CD: %.12f' % avd_cd)
logger.info('HD: %.12f' % avd_hd)
logger.info(f'class: x: {prediction1.max(dim=1)[1].item()}, ref: {prediction.max(dim=1)[1].item()}')

