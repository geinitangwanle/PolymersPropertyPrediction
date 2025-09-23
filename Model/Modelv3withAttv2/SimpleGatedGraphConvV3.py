import math
import torch
from torch_geometric.nn.conv import MessagePassing

class SimpleGatedGraphConv(MessagePassing):

    def __init__(self,
                 out_channels, #把输入节点特征映射到 out_channels 维度，并在所有消息传递、GRU 更新中保持这个维度
                 num_layers, # 本层内部要重复多少次“消息聚合 + GRU 更新”
                 num_edge_types, # 边类型的数量（比如化学键类型：单键、双键、三键、芳香键）
                 num_node_types, # 节点类型的数量（原子类型）
                 aggr, # 邻居消息的聚合方式，传给 MessagePassing 父类
                 edge_in_size, # 边特征 edge_attr 的维度
                 bias=True, # 是否在 GRUCell 里使用偏置。
                 use_nodetype_coeffs=False,# 是否启用“节点类型缩放”,默认为False
                 use_jumping_knowledge=False,# 是否保存每次迭代的中间结果。
                 use_bias_for_update=False, # 是否在聚合结果上加一个额外的可学习偏置向量
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
        # 学习型输入投影，当节点特征数量不等于out_channels时，添加一个线性变换层
        if node_in_dim is not None and node_in_dim != out_channels:
            self.in_proj = torch.nn.Linear(node_in_dim, out_channels)
            torch.nn.init.xavier_uniform_(self.in_proj.weight)
            torch.nn.init.zeros_(self.in_proj.bias)
        else:
            self.in_proj = None

        # 创建 num_layers 个 out_channels * out_channels 的权重矩阵
        self.weight = torch.nn.Parameter(torch.Tensor(num_layers, out_channels, out_channels))

        # 创建一个用于边特征线性变换的MLP
        self.nn = torch.nn.Sequential(torch.nn.Linear(edge_in_size, 128),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(128, self.out_channels * self.out_channels))

        #如果使用节点因子缩放，就添加一个num_layers * num_node_types 的权重矩阵，用于学习不同原子对于消息传递的重要性
        if self.use_nodetype_coeffs: 
            self.w_nodetypes = torch.nn.Parameter(
                torch.Tensor(num_layers, num_node_types))

        # 门控单元，用于防止多层卷积导致的信息丢失
        self.gru = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        # 为每一次内部消息传递迭代准备一层BatchNorm，防止特征分布出现偏移
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(out_channels) for _ in range(num_layers)])

        # 如果启用偏置，就额外弄一套参数
        if use_bias_for_update:
            self.bias_for_update = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_for_update', None)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        self.uniform(self.out_channels, self.weight) 
        for m in self.nn:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight); torch.nn.init.zeros_(m.bias) # 初始化边的线性变换权重
        if self.use_nodetype_coeffs:
            torch.nn.init.xavier_uniform_(self.w_nodetypes) # 如果使用节点因子缩放，就使用Xavier初始化
        if self.use_bias_for_update:
            torch.nn.init.zeros_(self.bias_for_update) # 如果使用偏置，就初始化为0
        self.gru.reset_parameters()
    
    def uniform(self, size, tensor): # 初始化权重
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        h = x if x.dim() == 2 else x.unsqueeze(-1) # 如果输入的节点特征是二维，就保持不变，否则增加一个维度
        if self.in_proj is not None:
            h = self.in_proj(h)
        elif h.size(1) != self.out_channels:
            # 兜底：不建议走这里，尽量在 __init__ 配置好 node_in_dim
            if h.size(1) > self.out_channels:
                raise ValueError("input dim > out_channels")
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        # 2) 边→矩阵
        edge_mat = None
        if edge_attr is not None:
            edge_mat = self.nn(edge_attr).view(-1, self.out_channels, self.out_channels)  # [E, C, C]
        
        ms = []
        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i]) # 对节点特征做线性变换

            if self.use_nodetype_coeffs: # 原子类型特征必须是one-hot
                # x 的前 num_node_types 维是 one-hot：
                x_onehot = x[:, :self.num_node_types]            # [N, T]
                lamb = x_onehot @ self.w_nodetypes[i].unsqueeze(1)  # [N, 1]
                m *= torch.sigmoid(lamb)
            
            # 调用消息传递做特征聚合
            if edge_attr is not None:
                m_new = self.propagate(edge_index, x=m, weight=edge_mat, edge_weight=edge_weight) #如果有边特征，就进行节点特征和边特征的联合汇聚
            else:
                m_new = self.propagate(edge_index, x=m, weight=None, edge_weight=edge_weight) #如果没有边特征，就只进行节点特征汇聚
            
            m_new = self.batch_norms[i](m_new)
            # 这个暂时不知道有什么用
            if self.use_jumping_knowledge:
                ms.append(m_new)  # BN →（可选 JK 收集）→ GRU
            h = self.gru(m_new, h)

        if self.use_jumping_knowledge:
            out = h, ms
        else:
            out = h
        return out
    
    def message(self, x_j, weight=None, edge_weight=None): 
        msg = x_j
        if weight is not None:
            msg = torch.matmul(msg.unsqueeze(1), weight).squeeze(1)  # [E, C]
        if edge_weight is not None:
            # edge_weight: [E] or [E,1]
            msg = msg * edge_weight.view(-1, 1)
        return msg
    
    def update(self, aggr_out): 
        if self.use_bias_for_update:
            aggr_out = aggr_out + self.bias_for_update
        return aggr_out