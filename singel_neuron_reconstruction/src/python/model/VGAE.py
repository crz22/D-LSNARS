import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from check_utils import find_link


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.uniform_(self.bias)

    def forward(self, features, adjacency):
        support = torch.mm(features, self.weight).cuda()  # 爱因斯坦求和约定,用于计算带有batch的矩阵乘法
        # print("sup: ",support)
        output = torch.spmm(adjacency, support)
        # print("out: ",output)
        if self.use_bias:
            return output + self.bias
        return output

    def __repr__(self):
        return (self.__class__.__name__ + ' ('
                + str(self.in_features) + ' -> '
                + str(self.out_features) + ')')


class MY_GCN(nn.Module):
    def __init__(self, in_features=3):
        super(MY_GCN, self).__init__()
        self.in_features = in_features

        self.gcn1 = GraphConvolution(self.in_features, 16)
        self.bn1 = nn.BatchNorm1d(16, track_running_stats=False)
        self.gcn2 = GraphConvolution(16, 32)
        self.bn2 = nn.BatchNorm1d(32, track_running_stats=False)
        self.gcn3 = GraphConvolution(32, 64)
        self.bn3 = nn.BatchNorm1d(64, track_running_stats=False)
        self.gcn4 = GraphConvolution(64, 128)
        self.bn4 = nn.BatchNorm1d(128, track_running_stats=False)
        self.gcn5 = GraphConvolution(128, 64)
        self.bn5 = nn.BatchNorm1d(64, track_running_stats=False)
        self.gcn6 = GraphConvolution(64, 32)
        self.gcn7 = GraphConvolution(64, 32)

    def forward(self, x, adjacency):
        x = F.relu(self.bn1(self.gcn1(x, adjacency)))
        # print("layer1: ",x)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2(self.gcn2(x, adjacency)))
        # print("layer2: ", x)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.bn3(self.gcn3(x, adjacency)))
        # print("layer3: ", x)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.gcn4(x, adjacency)))
        # print("layer4: ", x)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.bn5(self.gcn5(x, adjacency)))
        z_mean = self.gcn6(x, adjacency)
        z_std = self.gcn7(x, adjacency)
        return z_mean, z_std


class MY_GCN_1(nn.Module):
    def __init__(self, in_features=3):
        super(MY_GCN_1, self).__init__()
        self.in_features = in_features

        self.gcn1 = GraphConvolution(self.in_features, 16)
        self.bn1 = nn.BatchNorm1d(16, track_running_stats=False)
        self.gcn2 = GraphConvolution(16, 32)
        self.bn2 = nn.BatchNorm1d(32, track_running_stats=False)
        self.gcn3 = GraphConvolution(32, 64)
        self.bn3 = nn.BatchNorm1d(64, track_running_stats=False)
        self.gcn4 = GraphConvolution(64, 128)
        self.bn4 = nn.BatchNorm1d(128, track_running_stats=False)
        self.gcn5 = GraphConvolution(128, 256)
        self.bn5 = nn.BatchNorm1d(256, track_running_stats=False)
        self.gcn6 = GraphConvolution(256, 512)
        self.bn6 = nn.BatchNorm1d(512, track_running_stats=False)

        self.fc1 = nn.Sequential(nn.Linear(512, 256),
                                 nn.BatchNorm1d(256, track_running_stats=False))
        self.fc2 = nn.Sequential(nn.Linear(256, 128),
                                 nn.BatchNorm1d(128, track_running_stats=False))
        self.fc3 = nn.Sequential(nn.Linear(128, 64),
                                 nn.BatchNorm1d(64, track_running_stats=False))
        self.fc4 = nn.Sequential(nn.Linear(64, 5))
        # init.normal_(self.fc.wight, 0, math.sqrt(2. / 2))

    def forward(self, x, adjacency):
        x = F.relu(self.bn1(self.gcn1(x, adjacency)))
        # print("layer1: ",x)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2(self.gcn2(x, adjacency)))
        # print("layer2: ", x)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.bn3(self.gcn3(x, adjacency)))
        # print("layer3: ", x)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.gcn4(x, adjacency)))
        # print("layer4: ", x)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.bn5(self.gcn5(x, adjacency)))
        x = F.relu(self.bn6(self.gcn6(x, adjacency)))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = F.log_softmax(x, dim=1)
        return x
        # x = F.log_softmax(x, dim=1)
        # return x


class Decoder(nn.Module):
    def forward(self, z):
        # 解码过程，就是将得到的特征进行内积，得到一个点与点之间的关系，即adj
        # z = torch.sigmoid(z)
        adj = torch.mm(z, z.t())
        # adj = torch.sigmoid(adj)
        # adj = F.softmax(adj, dim=1)
        return adj


class GAE(nn.Module):
    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder  # 这里encoder的设置可以就是普通的GCN或者其他模型
        self.decoder = Decoder() if decoder is None else decoder  # decoder是上面的class

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_index):
        z_mean, z_std = self.encoder(x, edge_index)  # 特征提取网络最后的输出
        z = self.reparameterize(z_mean, z_std)
        reconstruction = self.decoder(z)  # 得到重建的adj
        return reconstruction, z_mean, z_std


def Rec_adj(adj_recon, adj_mask, adj_org, swc, t=0.5):
    """
    :param adj_org: 原始重建的连接关系
    :param adj_recon: 网络输出
    :param adj_mask: 已知有可能的连接
    :param t:  判断是否连接的阈值
    :param swc: swc
    :return: 邻接矩阵
    """

    adj_recover = adj_recon
    top_k, top_k_idx = torch.topk(adj_recover, k=5, dim=1)
    # adj_recover = torch.zeros_like(adj_recover)
    # for i in range(adj_recover.size(0)):
    #     adj_recover[i, top_k_idx[i]] = 1

    adj_recover = (adj_recover * adj_mask).float()
    adj_recover = torch.gt(adj_recover, t).float()

    adj_recover = adj_recover.to(torch.int)
    # adj_recover = adj_recover | adj_recover.t()

    adj_org = adj_org.to(torch.int)
    adj_x = adj_recover - (adj_recover & adj_org)
    adj_and = adj_recover & adj_org
    rem_x, rem_y = np.where(adj_x.cpu().numpy() == 1)

    for i in range(rem_x.shape[0]):
        idx_x = rem_x[i]
        idx_y = rem_y[i]
        '''
        if boundary(swc[idx_x,2:5]) or boundary(swc[idx_y,2:5]):
            adj_recover[idx_x][idx_y] = 0
            continue
        '''
        if adj_and[idx_x].sum() < 2 :
            point1 = torch.from_numpy(swc[idx_x, 2:5]).unsqueeze(dim=0)
            point2 = torch.from_numpy(swc[idx_y, 2:5]).unsqueeze(dim=0)
            point_dist = torch.cdist(point1, point2)
            # if idx_x == 177:
            #     print(idx_x,idx_y,find_link(idx_x, idx_y, swc))
            if point_dist >= 15 or find_link(idx_x, idx_y, swc):
                adj_recover[idx_x][idx_y] = 0
            '''
            elif adj_and[idx_x].sum() == 2:
                point1 = torch.from_numpy(swc[idx_x, 2:5]).unsqueeze(dim=0)
                point2 = torch.from_numpy(swc[idx_y, 2:5]).unsqueeze(dim=0)
                point_dist = torch.cdist(point1, point2)
                if point_dist >= 10 or find_link(idx_x, idx_y, swc):
                    adj_recover[idx_x][idx_y] = 0
            '''
        else:
            adj_recover[idx_x][idx_y] = 0

    return adj_recover

def boundary(point):
    if point[0] <= 12 and point[0] >= 255-12:
        return True
    if point[1] <= 12 and point[1] >= 255-12:
        return True
    if point[2] <= 12 and point[2] >= 255-12:
        return True
    return False