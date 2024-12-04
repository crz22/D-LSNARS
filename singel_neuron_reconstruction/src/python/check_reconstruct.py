import argparse
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from model.VGAE import MY_GCN, GAE, Rec_adj
from check_utils import loadswc, create_node_from_swc, build_nodelist, compute_trees, remove_fork_less_x, saveswc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_edge(swc):
    swc_adj = np.zeros((swc.shape[0], swc.shape[0]))
    for i in range(swc.shape[0]):
        cur_point = swc[i, 0].astype(np.int32)
        par_point = swc[i, -1].astype(np.int32)
        if par_point == -1:
            swc_adj[cur_point - 1, cur_point - 1] = 1
        else:
            swc_adj[cur_point - 1, par_point - 1] = 1
            swc_adj[cur_point - 1, cur_point - 1] = 1
        cld_list = np.where(swc[:, -1] == cur_point)[0]
        for cld_idx in cld_list:
            cld_point = swc[cld_idx, 0].astype(np.int32)
            swc_adj[cur_point - 1, cld_point - 1] = 1
    return swc_adj


def point_distance(points1, points2, k, is_max=False):
    """
    :param points1: swc中的某个点的x，y，z坐标（1，3）
    :param points: adj中的所有点及其坐标（n，3）
    :param k: 返回距离最大/最小的索引值个数
    :param is_max: True 返回前k个最大索引，False 返回前k个最小索引
    :return:
    """
    if type(points1) == np.ndarray:
        points1 = torch.from_numpy(points1)
    if type(points2) == np.ndarray:
        points2 = torch.from_numpy(points2)
    dist = torch.cdist(points1, points2, p=2)
    # print(dist.shape)
    top_dist, k_nn = torch.topk(dist, k + 1, dim=1, largest=is_max)
    k_nn = k_nn.squeeze(0).detach().cpu().numpy()
    return k_nn


def get_mask_edge(swc):
    # 通过距离判断，获取可能存在联系关系的点
    # kpp是指某个点与所有点的距离中，最小的k个值的索引
    num_points = swc.shape[0]
    mask_edge = np.zeros((num_points, num_points))

    # Get k nearest neighbors for each point
    k = 8 
    if k+1 > swc.shape[0]:
        k = swc.shape[0] - 1

    k_neighbors = point_distance(swc[:, 2:5], swc[:, 2:5], k=k, is_max=False)
    # print(k_neighbors)
    # Set the mask edge matrix based on k nearest neighbors
    for i in range(num_points):
        mask_edge[i, k_neighbors[i][1:]] = 1
        mask_edge[k_neighbors[i][1:], i] = 1
    return mask_edge


def norm_feature(feature):
    feature_min = np.array([feature[:, 0].min(), feature[:, 1].min(), feature[:, 2].min()])
    feature_max = np.array([feature[:, 0].max(), feature[:, 1].max(), feature[:, 2].max()])
    feature = (feature - feature_min) / (feature_max - feature_min)
    # print("min,max: ", feature_min, feature_max)
    # print(feature_max == feature_min)
    x = feature_max == feature_min
    return feature


def swc_progress(swc):
    adj = get_edge(swc)
    feature = swc[:, 2:5]
    feature = norm_feature(feature)
    adj_mask = get_mask_edge(swc)
    return feature, adj, adj_mask


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def get_swc_from_edges(adj, adj_mask, adj_org, swc, save_path):
    check_swc = swc.copy()
    adj = torch.sigmoid(adj)
    adj_rec = Rec_adj(adj, adj_mask, adj_org, check_swc, t=0.4)
    recovered = adj_rec.cpu().detach().numpy()
    max_idx = np.where(recovered == 1)
    indices = np.column_stack((max_idx[0], max_idx[1]))
    node_list = create_node_from_swc(check_swc, indices)
    node_list = compute_trees(node_list)
    check_swc = build_nodelist(node_list)
    check_swc = remove_fork_less_x(check_swc, 1)
    saveswc(save_path, check_swc)
    save_temp_path = save_path + '.error_conect.swc'
    saveswc(save_temp_path, check_swc)


def error_reconstruct(swc_path):
    swc = loadswc(swc_path)
    GCN = MY_GCN().to(device)
    model = GAE(GCN).to(device)
    model_path = r'F:\neuron_reconstruction_system\D_LSNARS\singel_neuron_reconstruction\src\python\weight\VGAE.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        feature, adj, adj_mask = swc_progress(swc.copy())
        feature = torch.from_numpy(feature).to(device).float()
        adj_mask = torch.from_numpy(adj_mask).to(device).float()
        adj_org = torch.from_numpy(adj).to(device).float()
        # print(adj_mask)
        adj = sp.csr_matrix(adj)
        adj_norm = preprocess_graph(adj)
        adj_norm = adj_norm.to(device).float()
        recovered, _, _ = model(feature, adj_norm)
        # print(recovered)
        swc_save_path = swc_path
        get_swc_from_edges(recovered, adj_mask, adj_org, swc, swc_save_path)


if __name__ == '__main__':
    print('error_reconstruct...')
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default=r'', help='swc path')
    args = parser.parse_args()
    error_reconstruct(args.input_path)
    '''