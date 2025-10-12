# SimpleGatedGraphConv.py
import math
import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GraphNorm

class SimpleGatedGraphConv(MessagePassing):
    def __init__(self,
                 out_channels,
                 num_layers,
                 num_edge_types,
                 num_node_types,
                 aggr,
                 edge_in_size,
                 bias=True,
                 use_nodetype_coeffs=False,
                 use_jumping_knowledge=False,
                 use_bias_for_update=False,
                 node_in_dim=None,
                 **kwargs):
        super(SimpleGatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.num_node_types = num_node_types
        self.use_nodetype_coeffs = use_nodetype_coeffs
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_bias_for_update = use_bias_for_update

        # 输入投影（当节点特征维度 != out_channels）
        if node_in_dim is not None and node_in_dim != out_channels:
            self.in_proj = torch.nn.Linear(node_in_dim, out_channels)
            torch.nn.init.xavier_uniform_(self.in_proj.weight)
            torch.nn.init.zeros_(self.in_proj.bias)
        else:
            self.in_proj = None

        # 每步一个线性核 W_i
        self.weight = torch.nn.Parameter(torch.Tensor(num_layers, out_channels, out_channels))

        # ===（可选重路径，不推荐默认开启）===
        # 将边特征映射到权重矩阵 E×C×C，显存重：
        # self.edge2mat = torch.nn.Sequential(
        #     torch.nn.Linear(edge_in_size, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, out_channels * out_channels)
        # )

        # GRU 单元
        self.gru = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        # Pre-Norm：GraphNorm（需要 batch！）
        self.norms = torch.nn.ModuleList([GraphNorm(out_channels) for _ in range(num_layers)])
        # LayerScale（标量版）
        self.res_scales = torch.nn.Parameter(torch.ones(num_layers) * 1e-2)

        # FiLM 边调制（轻量稳妥）
        self.edge_film = torch.nn.Sequential(
            torch.nn.Linear(edge_in_size, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 2*out_channels)
        )

        # 节点类型缩放（可选）
        if self.use_nodetype_coeffs:
            self.w_nodetypes = torch.nn.Parameter(torch.Tensor(num_layers, num_node_types))

        # 额外偏置（可选）
        if use_bias_for_update:
            self.bias_for_update = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_for_update', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.uniform(self.out_channels, self.weight)
        # if hasattr(self, 'edge2mat'):
        #     for m in self.edge2mat:
        #         if isinstance(m, torch.nn.Linear):
            #         torch.nn.init.xavier_uniform_(m.weight); torch.nn.init.zeros_(m.bias)

        # 初始化 FiLM
        for m in self.edge_film:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight); torch.nn.init.zeros_(m.bias)

        if self.use_nodetype_coeffs:
            torch.nn.init.xavier_uniform_(self.w_nodetypes)

        if self.use_bias_for_update:
            torch.nn.init.zeros_(self.bias_for_update)

        self.gru.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None, batch=None):
        # ---- 准备节点态为 [N,C] ----
        if x.dim() == 2:
            h = x
        else:
            # 常见是 [N,C]，若不是，尽量 squeeze 到 [N,C]
            h = x.squeeze(-1)
        if self.in_proj is not None:
            h = self.in_proj(h)
        elif h.size(1) != self.out_channels:
            if h.size(1) > self.out_channels:
                raise ValueError("input dim > out_channels")
            # 左对齐 + 零填充（兜底）
            pad = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, pad], dim=1)

        if isinstance(self.norms[0], GraphNorm) and batch is None:
            raise ValueError("GraphNorm 需要 batch 张量，请从上层传入 batch。")

        ms = []
        for i in range(self.num_layers):
            # ---- Pre-Norm ----
            h_norm = self.norms[i](h, batch) if isinstance(self.norms[i], GraphNorm) else self.norms[i](h)

            # ---- 节点线性核 ----
            m = torch.matmul(h_norm, self.weight[i])  # [N,C]

            # ---- 节点类型缩放（可选）----
            if self.use_nodetype_coeffs:
                x_onehot = x[:, :self.num_node_types]                     # [N,T]
                lamb = x_onehot @ self.w_nodetypes[i].unsqueeze(1)        # [N,1]
                m = m * torch.sigmoid(lamb)

            # ---- 消息传递（FiLM 边调制 + 可选 edge_weight）----
            m_new = self.propagate(edge_index, x=m, edge_attr=edge_attr, edge_weight=edge_weight)

            # ---- GRU 更新 + LayerScale 残差缩放 ----
            h_tilde = self.gru(m_new, h)                     # [N,C]
            h = h + self.res_scales[i] * (h_tilde - h)

            if self.use_jumping_knowledge:
                ms.append(h)  # 收集更新后的表示更有意义

        return (h, ms) if self.use_jumping_knowledge else h

    # 注意：MessagePassing 会根据 message() 的形参名从 propagate() 的 kwargs 里取同名张量
    def message(self, x_j, edge_attr=None, edge_weight=None):
        msg = x_j  # [E,C]

        # ---- FiLM 边调制 ----
        if edge_attr is not None:
            gamma, beta = self.edge_film(edge_attr).chunk(2, dim=-1)  # [E,C], [E,C]
            msg = msg * (1 + torch.tanh(gamma)) + beta

        # ---- 标量边权（如距离权）----
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)

        # === 如果你要改用“边矩阵权重”，打开下面这段，并在 forward 里传 weight=... ===
        # def message(self, x_j, weight=None, ...):
        # if weight is not None:
        #     msg = torch.bmm(msg.unsqueeze(1), weight).squeeze(1)  # [E,C]

        return msg

    def update(self, aggr_out):
        if self.use_bias_for_update:
            aggr_out = aggr_out + self.bias_for_update
        return aggr_out
