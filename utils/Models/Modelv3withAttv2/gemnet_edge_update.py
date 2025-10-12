# gemnet_edge_update.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

def gaussian_rbf(x, centers, gamma):
    # x: [*, 1] or [*,] ; centers: [K], gamma: float
    x = x.view(-1, 1)
    return torch.exp(-gamma * (x - centers.view(1, -1)) ** 2)  # [*, K]

class GemNetEdgeUpdate(nn.Module):
    """
    基于三体 (k-i-j) 的边更新（简化 GemNet 风格）：
    对每条有向边 e: i->j，聚合来自 k in N(i)\{j} 的三元组特征，得到边级增强嵌入。
    输出：edge_feat_enhanced: [E, D_edge_out]
    """
    def __init__(self,
                 K_r=16, r_min=0.0, r_max=4.0,          # 径向 RBF
                 K_a=8,                                 # 角度 RBF (在 cosθ 空间上做 Gaussian)
                 mlp_hidden=64,
                 out_dim=32,                            # 生成的“角增强边嵌入”维度
                 aggr='add'):
        super().__init__()
        self.aggr = aggr
        # RBF centers & gamma
        self.register_buffer('r_centers', torch.linspace(r_min, r_max, K_r))
        dr = (r_max - r_min) / K_r
        self.r_gamma = 1.0 / (2 * (dr ** 2) + 1e-9)

        # 对 cos(theta) 做 Gaussian 展开：中心均匀采样在 [-1, 1]
        self.register_buffer('a_centers', torch.linspace(-1.0, 1.0, K_a))
        da = (2.0) / K_a
        self.a_gamma = 1.0 / (2 * (da ** 2) + 1e-9)

        in_dim = 2 * K_r + K_a  # rbf(d_ij) + rbf(d_ik) + rbf(cosθ)
        self.triplet_mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, out_dim),
        )
    
    @torch.no_grad()
    def _build_triplets(self, edge_index, num_nodes):
        """
        对每条边 e: i->j 找到所有 (k->i) 的边索引，返回展开后的 (edge_e, edge_ki, i, j, k)
        """
        row, col = edge_index  # row: src=i, col: dst=j
        E = edge_index.size(1)

        # 邻接的 "入边" 视图：按中心 i 把 (k->i) 组织成 CSR
        # adj[row=i, col=k] 里存的是 “入边(k->i)” 的边索引
        val = torch.arange(E, device=edge_index.device)
        adj_in = SparseTensor(row=col, col=row, value=val, sparse_sizes=(num_nodes, num_nodes))
        rowptr, col_k, e_ki = adj_in.csr()  # rowptr for i, col_k: k 节点索引, e_ki: 对应 (k->i) 的边 id

        # 对每条边 e: i->j，取中心 i 的所有入边 (k->i)
        i = row
        j = col
        # 每条边 e 的邻接数量 = in_degree(i)
        deg_i = (rowptr[i + 1] - rowptr[i]).to(torch.long)  # [E]
        # 展开：把每条边 e 按 deg_i[e] 重复，构造三元组列表
        edge_e_expanded = torch.repeat_interleave(torch.arange(E, device=edge_index.device), deg_i)
        # 对应的 (k->i) 边索引拼接在一起
        idx_start = rowptr[i]
        # 取每个 i 的入边切片 [rowptr[i], rowptr[i+1])，按 e 的顺序拼接
        slices = [torch.arange(idx_start[e], idx_start[e] + deg_i[e], device=edge_index.device) for e in range(E)]
        idx_flat = torch.cat(slices, dim=0) if len(slices) > 0 else torch.empty(0, dtype=torch.long, device=edge_index.device)
        e_ki_expanded = e_ki[idx_flat]    # (k->i) 边 id 序列
        k_nodes = col_k[idx_flat]         # k 节点序列

        # 从边 id 还原 (i->j) 的 i/j；(k->i) 的 k/i
        i_e = row[edge_e_expanded]   # 与 e 对齐的 i
        j_e = col[edge_e_expanded]   # 与 e 对齐的 j
        k_e = k_nodes                # 与 e 对齐的 k

        # 去除 k == j 的三元组（GemNet/DimeNet 中常见做法）
        mask = (k_e != j_e)
        return edge_e_expanded[mask], e_ki_expanded[mask], i_e[mask], j_e[mask], k_e[mask]

    def forward(self, pos, edge_index, edge_attr=None):
        """
        输入:
            pos: [N,3]
            edge_index: [2,E]
            edge_attr: [E,D] or None（此处只用来返回时拼接，不在本模块内部使用）
        输出:
            edge_attr_enh: [E, D_trip], 其中 D_trip=out_dim
        """
        num_nodes = pos.size(0)
        row, col = edge_index
        E = edge_index.size(1)

        # 1) 构造三元组 (e: i->j, ki: k->i)
        e_e, e_ki, i_e, j_e, k_e = self._build_triplets(edge_index, num_nodes)
        if e_e.numel() == 0:
            # 退化情况：没有三元组（极稀图），返回零向量
            return pos.new_zeros((E, self.triplet_mlp[-1].out_features))

        # 2) 计算距离与角度
        rij = pos[j_e] - pos[i_e]               # [T,3]
        rik = pos[k_e] - pos[i_e]               # [T,3]
        dij = rij.norm(dim=-1, keepdim=True)    # [T,1]
        dik = rik.norm(dim=-1, keepdim=True)    # [T,1]

        # 角度 cosθ = <rij, rik> / (||rij||*||rik||)
        # 加上 eps 防止除零
        eps = 1e-8
        cos_theta = (rij * rik).sum(-1, keepdim=True) / (dij * dik + eps)  # [T,1]
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        # 3) RBF 展开
        rbf_ij = gaussian_rbf(dij, self.r_centers, self.r_gamma)      # [T, K_r]
        rbf_ik = gaussian_rbf(dik, self.r_centers, self.r_gamma)      # [T, K_r]
        abf    = gaussian_rbf(cos_theta, self.a_centers, self.a_gamma) # [T, K_a]
        triplet_feat = torch.cat([rbf_ij, rbf_ik, abf], dim=-1)       # [T, 2*K_r + K_a]

        # 4) 三元组 MLP -> 对 k 聚合到“边 e 级别”
        triplet_emb = self.triplet_mlp(triplet_feat)                  # [T, out_dim]
        # 聚合索引：e_e 表示这些三元组属于哪条边 e
        D = triplet_emb.size(-1)
        edge_trip = pos.new_zeros((E, D))
        edge_trip.index_add_(0, e_e, triplet_emb)   # sum 聚合；也可换 mean（再除以计数）

        # 如需 mean 聚合（GemNet 部分实现是正规化的），可以除以计数：
        # counts = pos.new_zeros((E,1))
        # counts.index_add_(0, e_e, torch.ones_like(e_e, dtype=counts.dtype).unsqueeze(-1))
        # edge_trip = edge_trip / (counts.clamp_min(1.0))

        return edge_trip  # [E, out_dim]