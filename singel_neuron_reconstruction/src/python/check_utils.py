import math
import numpy as np
import os
import libtiff as TIFF

TRUE_TYPE = 11
FALSE_TYPE = 2


def loadswc(filepath):
    # load swc file as a N X 7 numpy array
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                # remove empty units
                if len(cells) != 9:
                    for kk in range(len(cells) - 1, -1, -1):
                        if cells[kk] == '':
                            cells.pop(kk)
                if len(cells) != 7:
                    for i in range(7, len(cells)):
                        # print(i)
                        cells.pop()
                if len(cells) == 7:
                    cells = [float(c) for c in cells]  # transform string to float
                    swc.append(cells[0:7])
    return np.array(swc)


def saveswc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]
    # print(filepath)
    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)


def read_tif(tif_path):
    images = []
    tif = TIFF.TIFF.open(tif_path, mode='r')
    for image in tif.iter_images():
        # image = np.flipud(image)
        images.append(image)
    return np.array(images)


def save_tif(img, path):
    # if type == 'img':
    tif = TIFF.TIFF.open(path, mode='w')
    num = img.shape[0]
    for i in range(num):
        # img[i] = np.flipud(img[i])
        tif.write_image(((img[i]).astype(np.uint8)), compression=None)
    tif.close()
    return


def find_child_idx(swc, idx0):  # 寻找当前节点的所有子节点
    IDX = []  # 节点的标号
    for i in range(swc.shape[0]):
        if swc[i, 6] == idx0:
            IDX.append(i)
    return IDX


def find_parent_idx(swc, idx6):
    IDX = []
    for i in range(swc.shape[0]):
        if swc[i, 0] == idx6:
            IDX.append(i)
    return IDX


def resample_swc(swc_path):
    swc = loadswc(swc_path)
    axis7 = np.zeros((swc.shape[0], 1))
    swc = np.concatenate((swc, axis7), axis=1)
    # print(swc[:, 7])
    swc_copy = swc.copy()

    # print(swc_copy)
    for i in range(swc.shape[0]):
        index0 = swc[i, 0]  # 节点标号
        index6 = swc[i, 6]  # 其父节点标号
        swc[i, 0] = i + 1  # 将节点标号按顺序重置
        # print(index0)
        # a7 = swc[i, 7]
        ch_index = find_child_idx(swc_copy, index0)
        # print(ch_index)
        for j in ch_index:
            swc[j, 6] = i + 1  # 修改其子节点对应父节点位置的标号
            swc[j, 7] = -1  # 标记为修改

        if swc[i, 6] != -1 and swc[i, 7] != -1:
            pa_index = find_parent_idx(swc_copy, index6)
            if len(pa_index) == 0:
                swc[i, 6] = -1
    return swc


def resample_swc_file(swc_file_path, save_path):  # 对文件夹里的所有文件进行节点修整
    swc_file_name = os.listdir(swc_file_path)
    for i, file in enumerate(swc_file_name):
        swc_path = os.path.join(swc_file_path, file)
        swc = resample_swc(swc_path)
        swc_save_path = os.path.join(save_path, file)
        saveswc(swc_save_path, swc)


def cut(swc_, x, y, z):
    swc_block = []
    j = 0
    list1 = []
    # print(swc_.shape[0], x, y, z)
    for i in range(swc_.shape[0]):
        # print(swc_[i, (2, 3, 4)])
        # if x + 256 >= swc_[i, 2] >= x:
        # if i==104 :
        #     print(swc_[104])
        if (x + 256 >= swc_[i, 2] >= x) and (y + 256 >= swc_[i, 3] >= y) and (z + 256 >= swc_[i, 4] >= z):
            list1.append(swc_[i, 0])
            # print(swc_[i, 0], swc_[i, 2], swc_[i, 3], swc_[i, 4])
            swc_block.append(swc_[i, :])
            swc_block[j][2] = swc_block[j][2] - x  # 将x坐标变换到0-256区间
            swc_block[j][3] = swc_block[j][3] - y  # 将y坐标变换到0-256区间
            swc_block[j][4] = swc_block[j][4] - z  # 将z坐标变换到0-256区间
            # swc_block[j][0] = j+1                  # 变换节点标号
            # swc_block[j][6] = j                    # 变换节点上一个节点标号
            j = j + 1
    if not swc_block:
        swc_block = [[0, 0, 0, 0, 0, 0, 0]]
    # print(list1)
    # print(j, len(swc_block))
    return np.array(swc_block)


# 用于将整个GT_swc文件按重建的tif块裁剪
def swc_cut(swc_path, save_path, tif_file_path):
    swc = loadswc(swc_path)  # 需要截取的swc文件位置
    # print(swc[:, (2, 3, 4)]) xyz坐标位置
    print(swc.shape[0])
    swc_cut_sum = 0
    files_name = os.listdir(tif_file_path)  # 文件中还包含其他文件
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, file in enumerate(files_name):
        # print(file)
        # if file != 'x16754_y15437_z3040.tif':
        #     continue
        tif_right_name = file.split('.')[0] + '.tif'
        swc_copy = swc.copy()
        if file == tif_right_name:  # 过滤.swc文件
            # print(file)
            # tif_files_name.append(file)
            # 提取文件名中的xyz坐标
            file = file.replace('.tif', '')
            tif_xyz = file.split('_')
            tif_x = float(tif_xyz[0].replace('x', ''))
            tif_y = float(tif_xyz[1].replace('y', ''))
            tif_z = float(tif_xyz[2].replace('z', ''))
            # print(tif_x, tif_y, tif_z)
            swc_block = cut(swc_copy, tif_x, tif_y, tif_z)
            swc_block_path = os.path.join(save_path, file + '.swc')
            # print(swc_block_path)
            # print(swc_block)
            # print(swc_block.shape[0])
            saveswc(swc_block_path, swc_block)
            swc_cut_sum += swc_block.shape[0]
    # print(tif_files_name)
    # print(swc_cut_sum)


def save_to_txt(data, file_path):
    """
    将数据保存为文本文件

    参数:
    - data: 要保存的数据，可以是字符串或列表等
    - file_path: 要保存到的文件路径
    """
    try:
        with open(file_path, 'w') as file:
            # 如果数据是字符串，直接写入文件
            if isinstance(data, str):
                file.write(data)
            # 如果数据是列表，逐行写入文件
            elif isinstance(data, list):
                for item in data:
                    file.write(str(item) + '\n')
            else:
                print("Unsupported data type. Only strings and lists are supported.")
    except Exception as e:
        print(f"An error occurred: {e}")


# 将swc转换成swc_tree格式

class Graph:
    def __init__(self, num_of_nodes):
        self.num_of_nodes = num_of_nodes
        self.adjacency_list = {node: set() for node in range(num_of_nodes)}

    def add_edge(self, node1, child, parent):
        # print(node1)
        # print(node2)
        # node1 表示当前节点，node2表示其子节点，parent表示其父节点
        self.adjacency_list[node1].add((child, parent))

    def print_adj_list(self):
        for key in self.adjacency_list.keys():
            print(f'node {key} : {self.adjacency_list[key]}')

    def DFS(self, node0):
        # queue本质上是堆栈，用来存放需要进行遍历的数据
        # order里面存放的是具体的访问路径
        queue, order = [], []
        # 首先将初始遍历的节点放到queue中，表示将要从这个点开始遍历
        queue.append(node0)
        while queue:
            # 从queue中pop出点v，然后从v点开始遍历了，所以可以将这个点pop出，然后将其放入order中
            # 这里才是最有用的地方，pop（）表示弹出栈顶，由于下面的for循环不断的访问子节点，并将子节点压入堆栈，
            # 也就保证了每次的栈顶弹出的顺序是下面的节点
            node = queue.pop()
            if node != -1:
                order.append(node)
                for neighbor in self.adjacency_list[node]:
                    if neighbor not in order and neighbor not in queue and neighbor != -1:
                        queue.append(neighbor[0])
                    # print(neighbor[0])
        return order

    def BFS(self, node0):
        # queue本质上是堆栈，用来存放需要进行遍历的数据
        # order里面存放的是具体的访问路径
        queue, order = [], []
        queue.append(node0)
        order.append(node0)
        # 首先将初始遍历的节点放到queue中，表示将要从这个点开始遍历
        # 由于是广度优先，也就是先访问初始节点的所有的子节点，所以可以
        while queue:
            node = queue.pop(0)
            for neighbor in self.adjacency_list[node]:
                if neighbor not in order and neighbor[0] != -1:
                    queue.append(neighbor[0])
                    order.append(neighbor[0])
        return order

    def find_parent(self, node_id):
        parent_ids = []
        for node in self.adjacency_list.keys():
            # print(node)
            if node_id == node:
                parent_id = list(self.adjacency_list[node])
                # print(node)
                # print(parent_id)
                # print(self.adjacency_list[node])
                if parent_id:
                    parent_id = parent_id[0][1]
                    if parent_id != -1:
                        parent_ids.append(parent_id)
                        parent_ids.extend(self.find_parent(parent_id))
                break
        return parent_ids

    def find_neighbors(self, node_id):
        print(self.adjacency_list[node_id])
        x = self.adjacency_list[node_id]
        if x:
            print(len(x))

    def find_forks(self, node_id):
        child_idx = list()
        sub_node_list = list()
        node = self.adjacency_list[node_id]
        node_side_count = 0
        for neighbor in node:
            # print(neighbor[0])
            child_idx.append(neighbor[0])
        parent_idx = self.find_parent(node_id)
        sub_node_list.append(parent_idx)
        for child_id in child_idx:
            sub_node_list.append(self.DFS(child_id))
        if len(sub_node_list) >= 3:
            # print(node)
            # print(sub_node_list)
            if len(sub_node_list[1]) >= 3 and len(sub_node_list[2]) >= 3 and len(parent_idx) >= 1:
                print('fork-fork-1')
                return sub_node_list
            elif len(parent_idx) >= 3 and len(sub_node_list[1]) >= 2 and len(sub_node_list[2]) >= 2:
                print('fork-fork-2')
                return sub_node_list
            else:
                return None
        else:
            return None


# 用于寻找swc文件中存在的树结构
def swc_tree_create(swc):
    # swc = loadswc(swc_path)
    # print(swc)
    num_of_nodes = swc.shape[0]
    # print(swc.shape[0])
    # print(num_of_nodes)
    swc_tree = Graph(num_of_nodes + 1)
    for i in range(num_of_nodes):
        # print(node)
        node = swc[i, 0]
        node_idx0 = swc[i, 0]
        node_idx6 = swc[i, 6]
        child_idx = find_child_idx(swc, node_idx0)
        parent_idx = find_parent_idx(swc, node_idx6)
        if child_idx:
            for j in child_idx:
                swc_tree.add_edge(int(node), child=int(swc[j, 0]), parent=int(node_idx6))
        else:
            for j in parent_idx:
                swc_tree.add_edge(int(node), child=-1, parent=int(swc[j, 0]))
    return swc_tree


def swc_tree_find(swc):
    # swc = loadswc(swc_path)
    # print(swc_path)
    # print(swc.shape)
    swc_copy = np.copy(swc)
    # if swc.shape[0] == 0:
    #     swc = np.zeros((1, 7))
    #     saveswc(swc_path, swc)
    # print(swc.shape)
    swc_tree = swc_tree_create(swc_copy)
    axis7 = np.zeros((swc.shape[0], 1))
    # print(swc.shape)
    swc = np.concatenate((swc, axis7), axis=1)
    swc_tree_list = list()
    # swc_tree.print_adj_list()
    node_sum = 0
    for i in range(swc.shape[0]):
        # node 是节点的标号，i是节点在swc文件所在的位置
        node = i + 1
        if swc[i][7] != -1:
            child_tree_cut = list()
            child_idx = swc_tree.BFS(node)
            for j in child_idx:
                child_tree_cut.append(swc[j - 1])
                swc[j - 1][7] = -1

            parent_tree_cut = list()
            parent_idx = swc_tree.find_parent(node)
            parent_idx.reverse()
            for j in parent_idx:
                parent_tree_cut.append(swc[j - 1])
                swc[j - 1][7] = -1
            swc_tree_list.append(parent_tree_cut + child_tree_cut)
            node_sum += len(parent_idx + child_idx)
            # print(node, parent_idx)
            # print(node, parent_idx+child_idx)
    # print('cut_len:', node_sum)
    # print('swc_len:', swc.shape[0])
    return swc_tree_list


# 该函数用于对重建出来的swc文件进行裁剪，用于输入到之后的模型中
def resample_swc_(swc):
    # swc = loadswc(swc_path)
    if swc.shape[1] < 8:
        axis7 = np.zeros((swc.shape[0], 1))
        swc = np.concatenate((swc, axis7), axis=1)
    else:
        swc[:, 7] = 0
    # print(swc[:, 7])
    swc_copy = swc.copy()

    # print(swc_copy)
    for i in range(swc.shape[0]):
        index0 = swc[i, 0]  # 节点标号
        index6 = swc[i, 6]  # 其父节点标号
        swc[i, 0] = i + 1  # 将节点标号按顺序重置
        # print(index0)
        # a7 = swc[i, 7]
        ch_index = find_child_idx(swc_copy, index0)
        # print(ch_index)
        for j in ch_index:
            swc[j, 6] = i + 1  # 修改其子节点对应父节点位置的标号
            swc[j, 7] = -1  # 标记为修改

        if swc[i, 6] != -1 and swc[i, 7] != -1:
            pa_index = find_parent_idx(swc_copy, index6)
            if len(pa_index) == 0:
                swc[i, 6] = -1
    return swc


class Node(object):
    def __init__(self, position, radius, node_type=3):
        self.position = position
        self.radius = radius
        self.node_type = node_type
        self.nbr = []


def get_undiscover(dist):
    for i in range(dist.shape[0]):
        if dist[i] == 100000:
            return i
    return -1


def compute_trees(n0):
    n0_size = len(n0)
    treecnt = 0
    q = []
    n1 = []
    dist = np.ones([n0_size, 1], dtype=np.int32) * 100000
    nmap = np.ones([n0_size, 1], dtype=np.int32) * -1  # index in output tree n1
    parent = np.ones([n0_size, 1], dtype=np.int32) * -1  # parent index in current tree n0
    # print('Search for Soma')
    for i in range(n0_size):
        if n0[i].node_type == 1:
            q.append(i)
            dist[i] = 0
            nmap[i] = -1
            parent[i] = -1
    # BFS
    while len(q) > 0:
        curr = q.pop(0)
        # print('curr_node:', curr + 1)
        n = Node(n0[curr].position, n0[curr].radius, treecnt + 2)
        if parent[curr] >= 0:
            n.nbr.append((nmap[parent[curr]]))
            # print(len(n1) + 1, curr + 1, int(parent[curr] + 1))
        n1.append(n)
        nmap[curr] = len(n1)
        for j in range(len(n0[curr].nbr)):
            adj = n0[curr].nbr[j]
            if dist[adj] == 100000:
                dist[adj] = dist[curr] + 1
                parent[adj] = curr
                # print(adj,curr)
                q.append(adj)
    # print(len(n1))
    # print('xxxx')
    while get_undiscover(dist) >= 0:
        treecnt = treecnt + 1
        seed = get_undiscover(dist)
        dist[seed] = 0
        nmap[seed] = -1  ############
        parent[seed] = -1
        q.append(seed)
        while len(q) > 0:
            curr = q.pop(0)
            # print('curr_node:', curr + 1)
            n = Node(n0[curr].position, n0[curr].radius, treecnt + 2)
            if parent[curr] >= 0:
                n.nbr.append((nmap[parent[curr]]))
                # print(len(n1) + 1, curr + 1, int(parent[curr] + 1))
            n1.append(n)
            nmap[curr] = len(n1)
            for j in range(len(n0[curr].nbr)):
                adj = n0[curr].nbr[j]
                if dist[adj] == 100000:
                    dist[adj] = dist[curr] + 1
                    parent[adj] = curr
                    # print(adj, curr, nmap[parent[adj]])
                    q.append(adj)
    # print(len(n1))
    return n1


def build_nodelist(tree):
    _data = np.zeros((1, 7))
    cnt_recnodes = 0
    for i in range(len(tree)):
        if len(tree[i].nbr) == 0:
            cnt_recnodes += 1
            pid = -1
            new_node = np.asarray([cnt_recnodes,
                                   tree[i].node_type,
                                   tree[i].position[0],
                                   tree[i].position[1],
                                   tree[i].position[2],
                                   tree[i].radius,
                                   pid])
            _data = np.vstack((_data, new_node))

        else:
            # if len(tree[i].nbr) > 2:
            #    print(len(tree[i].nbr))
            for j in range(len(tree[i].nbr)):
                cnt_recnodes += 1
                pid = tree[i].nbr[j].squeeze()
                new_node = np.asarray([cnt_recnodes,
                                       tree[i].node_type,
                                       tree[i].position[0],
                                       tree[i].position[1],
                                       tree[i].position[2],
                                       tree[i].radius,
                                       pid])
                _data = np.vstack((_data, new_node))
    _data = _data[1:, :]
    return _data


def create_node_from_swc(swc, max_idx=None):
    swc_temp = swc.copy()
    node_list = []
    for i in range(swc_temp.shape[0]):
        node = Node(swc_temp[i, 2:5], swc_temp[i, 5], swc_temp[i, 1])
        if i == 0:
            node = Node(swc_temp[i, 2:5], swc_temp[i, 5], node_type=1)
        if max_idx is not None:
            idx = np.where(max_idx[:, 0] == i)
            # print(max_idx[idx][:, 1])
            if len(idx) >= 1:
                node.nbr = max_idx[idx][:, 1]
            # print(max_idx)
            # print(node.nbr)
        node_list.append(node)
    return node_list


def all_sample_joint(swc_root_path):
    file_list = os.listdir(swc_root_path)
    swc_first_path = os.path.join(swc_root_path, file_list[0])
    swc_all = loadswc(swc_first_path)
    for filename in file_list[1:]:
        swc_current_path = os.path.join(swc_root_path, filename)
        swc_current = loadswc(swc_current_path)
        swc_all = sample_joint(swc_all, swc_current)
    return swc_all


def sample_joint(swc1, swc2):
    swc1[0][6] = -1
    swc2[0][6] = -1
    for i in range(swc2.shape[0]):
        if swc2[i][6] >= swc2.shape[0]:
            swc2[i][6] = -1
        swc2[i][0] = swc2[i][0] + swc1.shape[0]
        if swc2[i][6] != -1:
            swc2[i][6] = swc2[i][6] + swc1.shape[0]
    return np.vstack((swc1, swc2))


def remove_false(false_swc_path, rec_swc):
    false_swc = loadswc(false_swc_path)
    # rec_swc = loadswc(rec_swc_path)
    for i in range(false_swc.shape[0]):
        f_x = round(false_swc[i][2], 3)
        f_y = round(false_swc[i][3], 3)
        f_z = round(false_swc[i][4], 3)  # 保留了小数后三位
        for j in range(rec_swc.shape[0]):
            r_x = round(rec_swc[j][2], 3)
            r_y = round(rec_swc[j][3], 3)
            r_z = round(rec_swc[j][4], 3)
            # print("f_swc:", f_x, f_y, f_z)
            # print("r_swc:", r_x, r_y, r_z)
            if f_x == r_x and f_y == r_y and f_z == r_z:
                rec_swc = np.delete(rec_swc, j, axis=0)
                # print('find it !')
                break
    return rec_swc


def change_all_type(swc_path, save_path, color):
    swc = loadswc(swc_path)
    for i in range(swc.shape[0]):
        if swc[i, 1] != color:
            swc[i, 1] = color
    print(swc.shape)
    if swc.shape[0] > 0:
        saveswc(save_path, swc)


def remove_fork_less_x(swc, len_cut):
    swc_tree = swc_tree_create(swc)
    rm_list = []
    for i in range(swc.shape[0]):
        node = swc[i, 0]
        child_idx = np.where(swc[:, 6] == node)[0]
        # print(child_idx.shape)
        if len(child_idx) >= 2:
            for idx in child_idx:
                child_node = int(swc[idx, 0])
                child_order = swc_tree.BFS(child_node)
                # print(child_order)
                if len(child_order) <= len_cut:
                    rm_list.extend(child_order)
    # print(rm_list)
    if rm_list:
        swc = np.delete(swc, np.array(rm_list) - 1, axis=0)
    swc = resample_swc_(swc)
    return swc

def find_link(point1, point2, swc):
    parent_idx_list = []
    parent_idx = point1
    parent_idx_list.append(parent_idx)
    while swc[parent_idx, 6] != -1:
        # print('swc.shape', swc.shape)
        parent_idx = np.where(swc[:, 0] == swc[parent_idx, 6])[0]
        # print('xxxx2')
        # print('parent_idx', parent_idx)
        # print('xxx: ', swc[parent_idx, 6])
        if np.isin(parent_idx, parent_idx_list):
            break
        parent_idx_list.extend(parent_idx)
    child_idx = np.where(swc[:, 6] == swc[parent_idx, 0])[0]
    # print(child_idx)
    child_idx_list = []
    while len(child_idx) > 0:
        # print('exit: ', np.isin(child_idx[0], child_idx_list))
        child_idx_refresh = []
        if np.isin(child_idx[0], child_idx_list):
            # print(child_idx_list)
            break
        child_idx_list.extend(child_idx)
        # print('swc.shape:', swc.shape)
        # print('child_idx_list:', child_idx_list)
        for idx in child_idx:
            next_idx = np.where(swc[:, 6] == swc[idx, 0])[0]
            # child_idx_list.extend(next_idx)
            child_idx_refresh.extend(next_idx)
        child_idx = child_idx_refresh

        # print('child_idx:', child_idx)
    parent_idx_list.extend(child_idx_list)
    # print(parent_idx_list)
    if np.isin(point2, parent_idx_list):
        return True
    return False

if __name__ == '__main__':
    '''
        根据重建裁剪出来的tif图像，裁剪出对应位置的GT图
    '''
    # dataset_root = r'I:\18454_reconstruction\dataset_version2'
    # GT_swc_root = r"I:\18454_reconstruction\GT"
    # for neuron_name in os.listdir(dataset_root):
    #     tif_root = os.path.join(dataset_root, neuron_name, 'Reconstruction')
    #     save_path = os.path.join(dataset_root, neuron_name, 'GT')
    #     GT_path = os.path.join(GT_swc_root, neuron_name + '.swc')
    #     swc_cut(swc_path=GT_path, tif_file_path=tif_root, save_path=save_path)

    '''
        将.tif_resampled.swc_removeTips.swc文件转移，并改变其颜色
    '''
    # data_root = r'I:\18454_reconstruction\grapn_nn_dataset1'
    # for filename in os.listdir(data_root):
    #     swc_root = os.path.join(data_root, filename, 'Reconstruction')
    #     save_root = os.path.join(data_root, filename, 'raw_sample')
    #     if not os.path.exists(save_root):
    #         os.makedirs(save_root)
    #     for file in os.listdir(swc_root):
    #         if file.endswith('.tif_resampled.swc_removeTips.swc'):
    #             swc_file_path = os.path.join(swc_root, file)
    #             save_path = os.path.join(save_root, file)
    #             # print(swc_file_path)
    #             change_all_type(swc_file_path, save_path, TRUE_TYPE)
    # print('finished!')

    """
        将swc文件转换成node类的列表
    """
    swc_path = r'I:\18454_reconstruction\grapn_nn_dataset1_test\train\18454_00004\raw_sample\x7856_y9674_z3929.tif_resampled.swc_removeTips.swc'
    swc = loadswc(swc_path)
    max_idx = np.ones((swc.shape[0], 2))
    node_list = create_node_from_swc(swc, max_idx)
    # print(node_list[0].node_type)
    print(node_list[0].nbr)
