import math

import torch
from torch_geometric.nn.conv import MessagePassing


class ModifiedGatedGraphConv(MessagePassing):

    def __init__(self,
                 out_channels, #把输入节点特征映射到 out_channels 维度，并在所有消息传递、GRU 更新中保持这个维度
                 num_layers, # 本层内部要重复多少次“消息聚合 + GRU 更新”
                 num_edge_types, # 边类型的数量（比如化学键类型：单键、双键、三键、芳香键）
                 num_node_types, # 节点类型的数量（原子类型）
                 aggr, # 邻居消息的聚合方式，传给 MessagePassing 父类
                 edge_in_size, # 边特征 edge_attr 的维度
                 bias=True, # 是否在 GRUCell 里使用偏置。
                 use_nodetype_coeffs=False,# 是否启用“节点类型缩放”
                 use_jumping_knowledge=False,# 是否保存每次迭代的中间结果。
                 use_bias_for_update=False, # 是否在聚合结果上加一个额外的可学习偏置向量
                 # use_edgeattr_data=True,
                 **kwargs):
        super(ModifiedGatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.num_node_types = num_node_types
        self.use_nodetype_coeffs = use_nodetype_coeffs
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_bias_for_update = use_bias_for_update
        # self.use_edgeattr_data = use_edgeattr_data

        self.weight = torch.nn.Parameter(
            # torch.Tensor(num_layers, num_edge_types, out_channels, out_channels)
            torch.Tensor(num_layers, out_channels, out_channels) # 创建 num_layers 个 out_channels * out_channels 的权重矩阵
        )

        # if use_edgeattr_data:
        self.nn = torch.nn.Sequential(torch.nn.Linear(edge_in_size, 128),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(128, self.out_channels * self.out_channels))
        # self.nn = torch.nn.Sequential(torch.nn.Linear(4, 128),
        #                                torch.nn.ReLU(),
        #                                torch.nn.Linear(128, self.out_channels * self.out_channels))

        if self.use_nodetype_coeffs: #如果使用节点因子缩放，就添加一个num_layers * num_node_types 的权重矩阵，用于学习不同原子对于消息传递的重要性
            self.w_nodetypes = torch.nn.Parameter(
                torch.Tensor(num_layers, num_node_types)
            )
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias) # 门控单元，用于防止多层卷积导致的信息丢失

        self.batch_norms = torch.nn.ModuleList( # 为每一次内部消息传递迭代准备一层BatchNorm，防止特征分布出现偏移
            [torch.nn.BatchNorm1d(out_channels) for _ in range(num_layers)]
        )

        if use_bias_for_update: #如果启用偏置，就额外弄一套参数
            self.bias_for_update = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_for_update', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.uniform(self.out_channels, self.weight)
        if self.use_nodetype_coeffs:
            torch.nn.init.xavier_normal_(self.w_nodetypes) # 使用 Xavier 初始化，让特征分布更加合理
            # self.uniform(self.num_node_types, self.w_nodetypes)
        if self.use_bias_for_update:
            torch.nn.init.zeros_(self.bias_for_update) # 初始化为0
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        h = x if x.dim() == 2 else x.unsqueeze(-1) # 如果输入的节点特征是二维，就保持不变，否则增加一个维度
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels: #如果输入的节点特征小于设置的维度，就补0
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        if edge_attr is not None: # 如果使用边特征，就对边特征做线性变换
            weight = self.nn(edge_attr).view(-1, self.out_channels, self.out_channels)

        ms = []
        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])

            if self.use_nodetype_coeffs: # 原子类型特征必须是one-hot
                lambda_mask = m.new_zeros(m.shape[0])
                for j in range(self.num_node_types):
                    mask = x[:, j] == 1
                    lambda_mask += mask * self.w_nodetypes[i, j]
                m *= torch.sigmoid(lambda_mask.unsqueeze(1))

            # if self.use_edgeattr_data:
            if edge_attr is not None: #调用消息传递做特征聚合
                m_new = self.propagate(edge_index, x=m, weight=weight)
            else:
                m_new = self.propagate(edge_index, x=m, weight=None) #只进行节点特征汇聚

            if self.use_jumping_knowledge:
                ms.append(m_new)  # last layer's output is excluded from JK output!

            m_new = self.batch_norms[i](m_new)

            h = self.rnn(m_new, h)

        if self.use_jumping_knowledge:
            out = h, ms
        else:
            out = h
        return out

    def message(self, x_j, weight): #edge_weight):#, pseudo): 这也没用上啊
        if weight is not None:
            return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
        else:
            return x_j

    def uniform(self, size, tensor): # 初始化权重
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def update(self, aggr_out): #这也没用啊
        if self.use_bias_for_update:
            aggr_out = aggr_out + self.bias_for_update
        return aggr_out

