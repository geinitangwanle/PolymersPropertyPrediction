import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
from .SimpleGatedGraphConv import SimpleGatedGraphConv
from torch_geometric.nn import global_mean_pool
from .geom_feats import GeometryFeaturizer

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
        num_convs=3, # 模型中的卷积块个数为3
        num_fc_layers=3,
        neighbors_aggr='add', # 节点特征聚合方式
        dropout_p=0.1,
        num_targets=1,
        # =========================== 新增接口 =====================
        geom_K=16, # RBF探针数量
        geom_rmax=4.0, # 设置化学键的最大键长
        concat_original_edge=True, # 得到新的边特征后，是否与原始边特征拼接
    ):
        super(GatedGCNModel, self).__init__()

        self.num_convs = num_convs
        self.layers_in_conv = layers_in_conv
        self.channels = channels
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_dropout = use_dropout
        self.pool_to_channels = nn.Linear(64, 128) # 线性层，用于将池化后的特征映射到128维
        self.geom_K = geom_K
        self.geom_rmax = geom_rmax
        self.concat_original_edge = concat_original_edge

        # === 几何特征编码器 ===
        self.geom = GeometryFeaturizer(K=geom_K, r_min=0.0, r_max=geom_rmax,concat_original=concat_original_edge)

        # 原始 edge_attr 4 维 + RBF K 维
        edge_in_dim = 4 + geom_K if concat_original_edge else 4


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

        self.mggc2 = SimpleGatedGraphConv(
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

        # 标准化，用于每个卷积块前
        self.batch_norms = nn.ModuleList(
            [torch.nn.BatchNorm1d(channels) for _ in range(num_convs)]
        )

        # 暂退
        self.dropout = nn.Dropout(p=dropout_p)

        # 构建多层线性层
        self.fc_layers = nn.ModuleList(
            self.make_fc_layers(num_fc_layers, num_targets=num_targets)
        )

        # 在线性层之前加一个标准化层
        self.pre_fc_batchnorm = torch.nn.BatchNorm1d(self.fc_layers[0].in_features)

        # 为每一个线性层之后加一个标准化层
        self.batch_norms_for_fc = nn.ModuleList(
            [
                torch.nn.BatchNorm1d(self.fc_layers[i + 1].in_features)
                for i in range(num_fc_layers - 1)
            ]
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

        # === 用几何编码器增强边特征 ===
        edge_attr_g = self.geom(pos, edge_index, edge_attr)  # [E, 4+K] or [E,K]

        x = self.mggc1(x, edge_index, edge_attr_g)
        x = self.batch_norms[0](x)
        x = F.relu(x)

        x = self.mggc2(x, edge_index, edge_attr_g)
        x = self.batch_norms[1](x)
        x = F.relu(x)

        x = self.mggc3(x, edge_index, edge_attr_g)
        x = self.batch_norms[2](x)
        x = F.relu(x)

        x_1 = self.set2set(x, batch)

        x_2 = global_mean_pool(x, batch) # 使用全局池化

        x = torch.cat([x_1, x_2], dim=1) # 拼接x1和x2

        x = self.pre_fc_batchnorm(x)

        for i, fc in enumerate(self.fc_layers):
            if self.use_dropout and i == 1: # 只在第一个线性层进行dropout
                x = self.dropout(x)
            x = fc(x)
            if i != len(self.fc_layers) - 1:
                x = self.batch_norms_for_fc[i](x)
                x = F.relu(x)

        return x