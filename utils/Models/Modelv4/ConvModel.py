# ConvModel.py 2025/10/05
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
from .SimpleGatedGraphConv import SimpleGatedGraphConv
from torch_geometric.nn import global_mean_pool
from .geom_feats import GeometryFeaturizer
from .gemnet_edge_update import GemNetEdgeUpdate
from torch_geometric.nn import GATv2Conv # 直接调用GATv2Conv

class GatedGCNModel(torch.nn.Module):
    def __init__(
        self,
        layers_in_conv=3, # 每个卷积块中图卷积层的个数为3，代表消息传递的次数
        channels=64, # 图卷积层中的隐藏维度
        use_nodetype_coeffs=True, 
        num_node_types=9, # 节点类型个数
        num_edge_types=4, # 边类型个数
        use_jumping_knowledge=False,
        embedding_size=64,
        use_bias_for_update=True,
        use_dropout=True,
        num_fc_layers=3,
        neighbors_aggr='add', # 节点特征聚合方式
        dropout_p=0.1,
        num_targets=1,
        # =========================== 新增接口 =====================
        geom_K=16, # RBF探针数量
        geom_rmax=4.0, # 设置化学键的最大键长
        concat_original_edge=True, # 得到新的边特征后，是否与原始边特征拼接（是否使用pos信息）
        gem_out=32,
        heads=4, # GAT多头注意力机制的头数
    ):
        super(GatedGCNModel, self).__init__()
        self.layers_in_conv = layers_in_conv
        self.channels = channels
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_dropout = use_dropout
        self.geom_K = geom_K
        self.geom_rmax = geom_rmax
        self.concat_original_edge = concat_original_edge

        # === 几何特征编码器 ===
        self.geom = GeometryFeaturizer(K=geom_K, r_min=0.0, r_max=geom_rmax,concat_original=concat_original_edge) # 边RBF

        self.gem_edge = GemNetEdgeUpdate(K_r=geom_K, r_min=0.0, r_max=geom_rmax, # 角度
                                         K_a=8, mlp_hidden=64, out_dim=gem_out, aggr='add')

        # 原始4 + 距离K + 角增强 gem_out
        edge_base = (4 if concat_original_edge else 0)
        edge_in_dim = edge_base + geom_K + gem_out


        self.mggc1 = SimpleGatedGraphConv(
            out_channels=channels,
            num_layers=layers_in_conv,
            num_edge_types=num_edge_types,
            num_node_types=num_node_types,
            aggr=neighbors_aggr,
            edge_in_size=edge_in_dim,
            use_nodetype_coeffs=False,
            use_jumping_knowledge=False,
            use_bias_for_update=True,
            node_in_dim=4,
        )

        assert channels % heads == 0, "channels 必须能被 heads 整除（concat=True）"
        self.norm_gat2 = nn.LayerNorm(channels)  # 保留 GAT 的 Pre-Norm
        self.ls_gat2   = nn.Parameter(torch.tensor(1e-2))  # LayerScale

        self.gat2 = GATv2Conv(
            in_channels=channels,
            out_channels=channels // heads,  # concat=True 时 head*out = channels
            heads=heads,
            concat=True,
            edge_dim=channels,            # 关键：把 (距离/角度/原始边) 送进注意力
            dropout=0.1,                     # 注意力 dropout
            add_self_loops=True
        )

        self.mggc3 = SimpleGatedGraphConv(
            out_channels=channels,
            num_layers=layers_in_conv,
            num_edge_types=num_edge_types,
            num_node_types=num_node_types,
            aggr=neighbors_aggr,
            edge_in_size=edge_in_dim,
            use_nodetype_coeffs=False,
            use_jumping_knowledge=False,
            use_bias_for_update=True,
            node_in_dim=channels,
        )

        #set2set全局池化层，使用了LSTM
        self.set2set = pyg.nn.Set2Set(channels, processing_steps=5, num_layers=2)

        # 暂退
        self.dropout = nn.Dropout(p=dropout_p)

        # 构建多层线性层
        self.fc_layers = nn.ModuleList(
            self.make_fc_layers(num_fc_layers, num_targets=num_targets)
        )

        # 读出前统一做一次 LayerNorm（图级向量更稳）
        self.readout_norm = nn.LayerNorm(3 * channels)

        # 简洁 MLP 头（无 BN，更适合小 batch 回归）
        hidden = max(3 * channels // 2, 64)
        self.head = nn.Sequential(
            nn.Linear(3 * channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout_p if use_dropout else 0.0),
            nn.Linear(hidden, num_targets)
        )

        # === 新增：边特征投影 + 门控（送入注意力）===
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_in_dim, channels),
            nn.Sigmoid()
        )


    # 用于构建线性层的功能函数
    def make_fc_layers(self, num_fc_layers, num_targets):
        fc_layers = []
        in_channels = 3 * self.channels
        for i in range(num_fc_layers):
            out_channels = num_targets if i == num_fc_layers - 1 else max(in_channels // 2, 8)
            fc_layers.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels
        return fc_layers
    
    
    

    # 前向传播部分
    def forward(self, data):
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch

        # (1) 边特征
        edge_rbf  = self.geom(pos, edge_index, edge_attr)
        edge_trip = self.gem_edge(pos, edge_index)
        edge_attr_all = torch.cat([edge_rbf, edge_trip], dim=-1)

        # Block-1：Gated（块内已有 GraphNorm + GRU + LS）
        x = self.mggc1(x, edge_index, edge_attr_all, batch=batch)
        x = F.relu(x)

        # Block-2：GATv2（Pre-LN + 残差 with LayerScale）
        e_proj  = self.edge_proj(edge_attr_all)
        e_gate  = self.edge_gate(edge_attr_all)
        e_final = e_proj * e_gate
        x_in = x
        x = self.gat2(self.norm_gat2(x), edge_index, e_final)
        x = x_in + self.ls_gat2 * x
        x = F.relu(x)

        # Block-3：Gated（注意要把输出写回 x）
        x = self.mggc3(x, edge_index, edge_attr_all, batch=batch)
        x = F.relu(x)

        # 读出
        x_s2s = self.set2set(x, batch)          # [B, 2C]
        x_mean = global_mean_pool(x, batch)     # [B, C]
        x = torch.cat([x_s2s, x_mean], dim=1)   # [B, 3C]

        x = self.readout_norm(x)

        for i, fc in enumerate(self.fc_layers):
            if self.use_dropout and i == 0: # 只在第一个线性层进行dropout
                x = self.dropout(x)
            x = fc(x)
            if i != len(self.fc_layers) - 1: # 最后一个线性层不激活
                x = F.relu(x)

        return x