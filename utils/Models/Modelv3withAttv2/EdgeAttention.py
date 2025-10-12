# edge_attention.py
import torch
import torch.nn as nn
from torch_geometric.utils import softmax

class EdgeAttention(nn.Module):
    """
    计算每条有向边 i->j 的注意力权重 alpha_ij ∈ (0,1)，并在每个目标节点 i 的邻域内做 softmax 归一化。
    输入:
      x: [N, C]     节点特征
      edge_index: [2, E]
      edge_attr: [E, D_e]  (几何增强后的边特征: 距离RBF/角三元组/原始边等)
    输出:
      alpha: [E]    每条边的权重（按目标 i 归一化）
    """
    def __init__(self, node_dim, edge_dim, hidden=64, dropout_p=0.1, temperature=1.0):
        super().__init__()
        self.lin_i = nn.Linear(node_dim, hidden, bias=False)
        self.lin_j = nn.Linear(node_dim, hidden, bias=False)
        self.lin_e = nn.Linear(edge_dim, hidden, bias=False)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.temperature = temperature

        nn.init.xavier_uniform_(self.lin_i.weight)
        nn.init.xavier_uniform_(self.lin_j.weight)
        nn.init.xavier_uniform_(self.lin_e.weight)
        nn.init.xavier_uniform_(self.score.weight)

    def forward(self, x, edge_index, edge_attr):
        j, i = edge_index[0], edge_index[1]              # 注意：MessagePassing默认聚合到 i（dst）
        h = torch.tanh(self.lin_i(x[i]) + self.lin_j(x[j]) + self.lin_e(edge_attr))  # [E, hidden]
        h = self.dropout(h)
        logits = self.score(h).squeeze(-1) / self.temperature                          # [E]
        alpha = softmax(logits, i, num_nodes=x.size(0))                                # 按目标节点 i 归一化
        return alpha
