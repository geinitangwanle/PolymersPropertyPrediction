# geom_feats.py
import torch
import torch.nn as nn

class GeometryFeaturizer(nn.Module):
    """
    把边的欧式距离用 RBF 展开，并与原始 edge_attr 拼接。
    输出维度 = (orig_dim if concat_original else 0) + K
    """
    def __init__(self, K=16, r_min=0.0, r_max=4.0, concat_original=True):
        super().__init__()
        self.K = K 
        self.r_min = r_min
        self.r_max = r_max
        self.concat_original = concat_original

        # 在(r_min, r_max)均匀放置 K 个中心
        centers = torch.linspace(r_min, r_max, K).view(1, K)  # [1,K]
        self.register_buffer("centers", centers)
        # RBF 宽度（可按经验设为间距）
        delta = (r_max - r_min) / K # K点之间的间距
        gamma = 1.0 / (2 * (delta ** 2) + 1e-9) # K点周围的宽度
        self.register_buffer("gamma", torch.tensor(gamma)) 

    @torch.no_grad()
    # 计算原子与原子间的相对坐标，确保平移和旋转不变性
    def _pair_dist(self, pos, edge_index):
        row, col = edge_index
        diff = pos[row] - pos[col]        # [E,3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [E,1] # 将distance=(x, y, z)转换成标量
        return dist
    
    def forward(self, pos, edge_index, edge_attr=None):
        """
        pos: [N,3], edge_index: [2,E], edge_attr: [E,D] or None
        return: new_edge_attr: [E, D+K] 或 [E,K]
        """
        dist = self._pair_dist(pos, edge_index)    # [E,1]
        # RBF: exp(-gamma * (d - mu)^2)
        # dist: [E,1], centers: [1,K] -> broadcast -> [E,K]
        rbf = torch.exp(-self.gamma * (dist - self.centers) ** 2)

        if self.concat_original and edge_attr is not None:
            return torch.cat([edge_attr, rbf], dim=-1)  # [E, D+K] 如果有边的特征，就做拼接
        return rbf  # 只有几何