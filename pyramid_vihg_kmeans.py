"""
Pyramid Vision Hypergraph Network (PViHG) with K-means based hypergraph construction.
Based on pyramid_vig_new.py structure, replaces FCM with fast K-means clustering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def pairwise_distance(x):
    """x: (B, N, C) -> (B, N, N) pairwise squared Euclidean distance."""
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(x ** 2, dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def act_layer(act, inplace=False, neg_slope=0.2):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'prelu':
        return nn.PReLU(init=neg_slope)
    else:
        raise NotImplementedError(f'Activation {act} not implemented')


def norm_layer(norm, nc):
    norm = norm.lower()
    if norm == 'batch':
        return nn.BatchNorm2d(nc)
    elif norm == 'instance':
        return nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(f'Norm {norm} not implemented')


# ----------------------------------------------------------------------
# K-means clustering (fully vectorized, with empty cluster handling)
# ----------------------------------------------------------------------
def kmeans_clustering(x, n_clusters, max_iter=10, epsilon=1e-4):
    """
    Batched K-means clustering.
    x: (B, N, C) input features.
    n_clusters: number of clusters.
    max_iter: maximum iterations.
    epsilon: convergence threshold (fraction of points reassigned).
    Returns:
        assignments: (B, N) int64 labels (0..n_clusters-1)
        centers: (B, n_clusters, C)
    """
    B, N, C = x.shape
    device = x.device
    dtype = x.dtype

    # Use float32 for numerical stability
    x = x.float()

    # Initialize centers by randomly selecting points from each batch
    indices = torch.randint(0, N, (B, n_clusters), device=device)
    centers = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, C))

    for _ in range(max_iter):
        # Compute distances to centers: (B, N, n_clusters)
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)                     # (B, N, 1)
        c_norm_sq = (centers ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, n_clusters)
        cross = torch.bmm(x, centers.transpose(1, 2))                      # (B, N, n_clusters)
        dist = x_norm_sq + c_norm_sq - 2 * cross                           # (B, N, n_clusters)

        new_assignments = dist.argmin(dim=-1)                              # (B, N)

        # Check convergence (fraction of points changed)
        if _ > 0:
            changed = (new_assignments != old_assignments).float().mean()
            if changed < epsilon:
                break
        old_assignments = new_assignments

        # --- Vectorized center update using scatter_add ---
        flat_assign = new_assignments.reshape(-1)                          # (B*N,)
        flat_x = x.reshape(B * N, C)                                       # (B*N, C)

        batch_idx = torch.arange(B, device=device).repeat_interleave(N)    # (B*N,)
        global_cluster = batch_idx * n_clusters + flat_assign              # (B*N,)

        cluster_sum = torch.zeros(B * n_clusters, C, device=device, dtype=torch.float32)
        cluster_sum.scatter_add_(0, global_cluster.unsqueeze(1).expand(-1, C), flat_x)

        cluster_count = torch.zeros(B * n_clusters, dtype=torch.long, device=device)
        ones = torch.ones_like(global_cluster)
        cluster_count.scatter_add_(0, global_cluster, ones)

        empty_mask = cluster_count == 0
        cluster_count_safe = cluster_count.clamp(min=1)
        new_centers_flat = cluster_sum / cluster_count_safe.unsqueeze(1).float()
        new_centers = new_centers_flat.reshape(B, n_clusters, C)

        # Reinitialize empty clusters with random points
        if empty_mask.any():
            empty_global = torch.where(empty_mask)[0]
            empty_batch = empty_global // n_clusters
            empty_cluster = empty_global % n_clusters
            rand_idx = torch.randint(0, N, (len(empty_global),), device=device)
            new_centers[empty_batch, empty_cluster, :] = x[empty_batch, rand_idx, :]

        centers = new_centers

    # Convert back to original dtype
    assignments = new_assignments.to(device=device, dtype=torch.long)
    centers = centers.to(dtype)
    return assignments, centers


def construct_hyperedges_kmeans(x, n_clusters, max_iter=10):
    """
    Build hypergraph using K-means clustering.
    Each cluster becomes a hyperedge containing its member points.
    Returns:
        hyperedge_matrix: (B, n_clusters, max_points) – node indices per hyperedge (padded with -1)
        point_hyperedge_index: (B, N, 1) – cluster index for each node
        centers: (B, C, n_clusters) – cluster centers
    """
    B, C, N, _ = x.shape
    device = x.device
    dtype = x.dtype

    # Prepare data (B, N, C)
    x_data = x.squeeze(-1).transpose(1, 2).contiguous()

    assignments, centers = kmeans_clustering(x_data, n_clusters, max_iter)  # (B,N), (B,n_clusters,C)

    # --- Build hyperedge_matrix using scatter (vectorized) ---
    flat_assign = assignments.reshape(-1)                                   # (B*N,)
    batch_idx = torch.arange(B, device=device).repeat_interleave(N)        # (B*N,)

    # Sort by cluster to group points of the same cluster together
    sorted_order = torch.argsort(flat_assign)
    sorted_assign = flat_assign[sorted_order]                               # (B*N,)
    sorted_batch = batch_idx[sorted_order]                                  # (B*N,)
    sorted_point = torch.arange(N, device=device).repeat(B)[sorted_order]  # (B*N,)

    # Find boundaries of each cluster
    unique_keys, counts = torch.unique_consecutive(sorted_assign, return_counts=True)
    max_points = counts.max().item()

    # Prepare linear indices for scatter
    offsets = torch.zeros_like(counts)
    offsets[1:] = counts.cumsum(0)[:-1]
    pos_in_cluster = torch.arange(len(sorted_assign), device=device) - offsets.repeat_interleave(counts)

    # Global linear index: batch * n_clusters * max_points + cluster * max_points + pos
    linear_idx = sorted_batch * n_clusters * max_points + sorted_assign * max_points + pos_in_cluster

    hyperedge_matrix = -torch.ones(B * n_clusters * max_points, dtype=torch.long, device=device)
    hyperedge_matrix[linear_idx] = sorted_point
    hyperedge_matrix = hyperedge_matrix.reshape(B, n_clusters, max_points)

    # point_hyperedge_index: each node belongs to exactly one cluster
    point_hyperedge_index = assignments.unsqueeze(-1)                       # (B, N, 1)

    # centers: (B, C, n_clusters)
    centers = centers.transpose(1, 2).contiguous()

    return hyperedge_matrix, point_hyperedge_index, centers


# ----------------------------------------------------------------------
# Hypergraph Convolution Layer (vectorized with scatter_add)
# ----------------------------------------------------------------------
class HypergraphConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, act='gelu', norm='batch', bias=True):
        super().__init__()
        self.node_to_hyper = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=bias),
            norm_layer(norm, in_channels),
            act_layer(act)
        )
        self.hyper_to_node = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            norm_layer(norm, out_channels)
        )
        self.eps = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, hyperedge_matrix, point_hyperedge_index, centers=None):
        B, C, H, W = x.shape
        N = H * W
        n_clusters = hyperedge_matrix.size(1)
        device = x.device

        x_flat = x.reshape(B, C, N)                                   # (B, C, N)

        # ------------------------------------------------------------------
        # Node -> hyperedge aggregation
        # ------------------------------------------------------------------
        mask_h = hyperedge_matrix != -1                               # (B, n_clusters, max_points)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(mask_h)[mask_h]
        cluster_idx = torch.arange(n_clusters, device=device).view(1, n_clusters, 1).expand_as(mask_h)[mask_h]
        node_idx = hyperedge_matrix[mask_h]                           # (V,)

        global_cluster = batch_idx * n_clusters + cluster_idx         # (V,)
        global_node = batch_idx * N + node_idx                        # (V,)

        x_flat_2d = x_flat.permute(0, 2, 1).reshape(B * N, C)         # (B*N, C)
        node_feats = x_flat_2d[global_node, :]                        # (V, C)

        total_clusters = B * n_clusters
        hyperedge_sum = torch.zeros(total_clusters, C, device=device, dtype=x.dtype)
        hyperedge_sum.scatter_add_(0, global_cluster.unsqueeze(1).expand(-1, C), node_feats)

        hyperedge_cnt = torch.zeros(total_clusters, dtype=torch.long, device=device)
        ones = torch.ones_like(global_cluster)
        hyperedge_cnt.scatter_add_(0, global_cluster, ones)

        hyperedge_cnt = hyperedge_cnt.clamp(min=1)
        hyperedge_feat = hyperedge_sum / hyperedge_cnt.unsqueeze(1).float()  # (total_clusters, C)
        hyperedge_feat = hyperedge_feat.reshape(B, n_clusters, C).permute(0, 2, 1)  # (B, C, n_clusters)
        hyperedge_feat = hyperedge_feat.unsqueeze(-1)                 # (B, C, n_clusters, 1)

        # Transform hyperedge features
        hyperedge_feat = self.node_to_hyper(hyperedge_feat)           # (B, C, n_clusters, 1)
        hyperedge_feat_sq = hyperedge_feat.squeeze(-1)                # (B, C, n_clusters)

        # ------------------------------------------------------------------
        # Hyperedge -> node aggregation
        # ------------------------------------------------------------------
        # point_hyperedge_index: (B, N, 1) -> squeeze to (B, N)
        assignments = point_hyperedge_index.squeeze(-1)               # (B, N)

        batch_idx_p = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1)  # (B*N,)
        node_idx_p = torch.arange(N, device=device).view(1, N).expand(B, N).reshape(-1)   # (B*N,)
        cluster_idx_p = assignments.reshape(-1)                       # (B*N,)

        global_node_p = batch_idx_p * N + node_idx_p                  # (B*N,)
        global_hyperedge_p = batch_idx_p * n_clusters + cluster_idx_p # (B*N,)

        hyperedge_feat_2d = hyperedge_feat_sq.permute(0, 2, 1).reshape(total_clusters, C).contiguous()
        hyperedge_feat_for_nodes = hyperedge_feat_2d[global_hyperedge_p, :]  # (B*N, C)

        total_nodes = B * N
        node_sum = torch.zeros(total_nodes, C, device=device, dtype=x.dtype)
        node_sum.scatter_add_(0, global_node_p.unsqueeze(1).expand(-1, C), hyperedge_feat_for_nodes)

        node_cnt = torch.zeros(total_nodes, dtype=torch.long, device=device)
        ones_p = torch.ones_like(global_node_p)
        node_cnt.scatter_add_(0, global_node_p, ones_p)

        node_cnt = node_cnt.clamp(min=1)
        node_aggr = node_sum / node_cnt.unsqueeze(1).float()          # (total_nodes, C)
        node_aggr = node_aggr.reshape(B, N, C).permute(0, 2, 1)      # (B, C, N)

        # GIN-style residual
        out = (1 + self.eps) * x_flat + node_aggr                     # (B, C, N)
        out = out.unsqueeze(-1)                                       # (B, C, N, 1)
        out = self.hyper_to_node(out)                                 # (B, out_channels, N, 1)
        return out.reshape(B, -1, H, W)


# ----------------------------------------------------------------------
# Grapher Block (with K-means hypergraph construction)
# ----------------------------------------------------------------------
class Grapher(nn.Module):
    def __init__(self, dim, n_clusters=25, dilation=1, conv='mr', act='gelu', norm='batch',
                 bias=True, stochastic=False, epsilon=0.0, reduce_ratio=1, n=196,
                 drop_path=0.0, relative_pos=False, kmeans_max_iter=10):
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans_max_iter = kmeans_max_iter
        self.conv_type = conv
        if conv == 'hypergraph':
            self.fc1 = nn.Sequential(
                nn.Conv2d(dim, dim, 1, bias=bias),
                norm_layer(norm, dim)
            )
            self.hypergraph_conv = HypergraphConv2d(dim, dim*2, act, norm, bias)
            self.fc2 = nn.Sequential(
                nn.Conv2d(dim*2, dim, 1, bias=bias),
                norm_layer(norm, dim)
            )
            self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        else:
            raise NotImplementedError("Only hypergraph convolution is supported in this version.")

    @torch.no_grad()
    def _build_hypergraph(self, x_flat):
        return construct_hyperedges_kmeans(x_flat, self.n_clusters, self.kmeans_max_iter)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H*W, 1)

        hyperedge_matrix, point_hyperedge_index, centers = self._build_hypergraph(x_flat)
        x = self.hypergraph_conv(x, hyperedge_matrix, point_hyperedge_index, centers)
        x = self.fc2(x)
        return self.drop_path(x) + shortcut


# ----------------------------------------------------------------------
# Feed-Forward Network
# ----------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


# ----------------------------------------------------------------------
# Stem (Patch Embedding)
# ----------------------------------------------------------------------
class Stem(nn.Module):
    """Overlapping patch embedding (2 downsampling stages)"""
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.convs(x)


# ----------------------------------------------------------------------
# Downsample
# ----------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.conv(x)


# ----------------------------------------------------------------------
# Main Model (Pyramid DeepGCN)
# ----------------------------------------------------------------------
class DeepGCN(nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        n_clusters = opt.k                     # number of hyperedges (clusters)
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        drop_path = opt.drop_path
        # Safely get kmeans_max_iter attribute (default to 10)
        kmeans_max_iter = getattr(opt, 'kmeans_max_iter', 10)

        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]            # from original, unused
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]

        # Optionally increase number of clusters across blocks (like dilation)
        # We'll keep it constant for simplicity, but can be modified.
        num_clusters = [int(x.item()) for x in torch.linspace(n_clusters, n_clusters, self.n_blocks)]

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        trunc_normal_(self.pos_embed, std=0.02)

        HW = (224 // 4) * (224 // 4)            # 56*56 = 3136

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    nn.Sequential(
                        Grapher(channels[i], n_clusters=num_clusters[idx], dilation=min(idx//4+1, 49//num_clusters[idx]),
                                conv=conv, act=act, norm=norm, bias=bias,
                                stochastic=stochastic, epsilon=epsilon,
                                reduce_ratio=reduce_ratios[i], n=HW,
                                drop_path=dpr[idx], relative_pos=True,
                                kmeans_max_iter=kmeans_max_iter),
                        FFN(channels[i], channels[i]*4, act=act, drop_path=dpr[idx])
                    )
                ]
                idx += 1

        self.prediction = nn.Sequential(
            nn.Conv2d(channels[-1], 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(opt.dropout),
            nn.Conv2d(1024, opt.n_classes, 1, bias=True)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        for blk in self.backbone:
            x = blk(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


# ----------------------------------------------------------------------
# Configuration and Model Registration
# ----------------------------------------------------------------------
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.convs.0', 'classifier': 'prediction.4',
        **kwargs
    }


default_cfgs = {
    'pvihg_kmeans_ti_224': _cfg(),
    'pvihg_kmeans_s_224': _cfg(),
    'pvihg_kmeans_m_224': _cfg(),
    'pvihg_kmeans_b_224': _cfg(crop_pct=0.95),
}


@register_model
def pvihg_kmeans_ti_224(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 25                         # number of clusters (hyperedges)
            self.conv = 'hypergraph'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]          # ti variant
            self.channels = [48, 96, 240, 384]
            self.n_classes = num_classes
            self.kmeans_max_iter = kwargs.get('kmeans_max_iter', 10)
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['pvihg_kmeans_ti_224']
    return model


@register_model
def pvihg_kmeans_s_224(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 25
            self.conv = 'hypergraph'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]
            self.channels = [80, 160, 400, 640]
            self.n_classes = num_classes
            self.kmeans_max_iter = kwargs.get('kmeans_max_iter', 10)
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['pvihg_kmeans_s_224']
    return model


@register_model
def pvihg_kmeans_m_224(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 25
            self.conv = 'hypergraph'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 16, 2]
            self.channels = [96, 192, 384, 768]
            self.n_classes = num_classes
            self.kmeans_max_iter = kwargs.get('kmeans_max_iter', 10)
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['pvihg_kmeans_m_224']
    return model


@register_model
def pvihg_kmeans_b_224(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 25
            self.conv = 'hypergraph'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 18, 2]
            self.channels = [128, 256, 512, 1024]
            self.n_classes = num_classes
            self.kmeans_max_iter = kwargs.get('kmeans_max_iter', 10)
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['pvihg_kmeans_b_224']
    return model