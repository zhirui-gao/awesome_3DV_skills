import torch
import math


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def chamfer_loss(points_src, points_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (torch tensor): source points
        normals_src (torch tensor): source normals
        points_tgt (torch tensor): target points
        normals_tgt (torch tensor): target normals
    '''
    dist_matrix = ((points_src.unsqueeze(2) - points_tgt.unsqueeze(1))**2).sum(-1)
    dist_complete = (dist_matrix.min(-1)[0]).mean(-1)
    dist_acc = (dist_matrix.min(-2)[0]).mean(-1)
    dist = ((dist_acc + dist_complete)/2).mean()
    return dist


def chamfer_loss_chunk_efficient(points_src, points_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    with less memory, 1, partition the points to chunks; 
    2, with no grad 
    Args:
        points_src (torch tensor): source points
        normals_src (torch tensor): source normals
        points_tgt (torch tensor): target points
        normals_tgt (torch tensor): target normals
    '''
    with torch.no_grad():
        B, N, _ = points_src.shape
        chunk_size = 1000
        G = points_tgt.shape[1]
        chunk_num = math.ceil(G/chunk_size)
        point_tgt_list = torch.chunk(points_tgt, chunk_num, 1)
        dist_complete_list = []
        dist_complete_index_list = []
        dist_acc_list = []
        for p_tgt in point_tgt_list:
            dist_matrix = ((points_src.unsqueeze(2) - p_tgt.unsqueeze(1)) ** 2).sum(-1)  # B N T
            rest = dist_matrix.min(-1)
            dist_complete_list.append(rest[0])  # B N 1
            dist_complete_index_list.append(rest[1])
            dist_acc_list.append((dist_matrix.min(-2)[1]))  # B 1 C
        target_closest_index = torch.stack(dist_acc_list).permute(1, 0, 2).reshape(B, -1)  # B T

        # R B N
        row_index = torch.stack(dist_complete_list).min(0)[1]
        BNR = torch.stack(dist_complete_index_list).permute(1,2,0).reshape(B*N, -1)  # B N R

        BN = BNR[torch.arange(B*N), row_index.reshape(-1)].reshape(B, N)

        src_cloest_index = row_index*chunk_size + BN
        src_cloest_point = index_points(points_tgt, src_cloest_index)
        target_closest_point = index_points(points_src, target_closest_index)
    dist_complete = ((points_src - src_cloest_point)**2).sum(-1).mean()
    dist_acc = ((points_tgt - target_closest_point) ** 2).sum(-1).mean()

    dist = ((dist_acc*1 + dist_complete*1)/2)
    return dist


if __name__=="__main__":
    points_src = torch.rand(16, 3000, 3)
    points_tgt = torch.rand(16, 6000, 3)
    # dist = chamfer_loss(points_src, points_tgt)
    dist = chamfer_loss_chunk_efficient(points_src, points_tgt)
    print(dist)
    
