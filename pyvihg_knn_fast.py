"""
Pyramid Vision Hypergraph Network (PViHG) with KNN-based hypergraph construction.
Based on pyramid_vig_new.py structure, but replaces FCM with fast KNN hypergraph building.
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
    """Compute pairwise squared Euclidean distance.
    x: (B, N, C) -> (B, N, N)
    """
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
# KNN hypergraph construction (fully vectorized)
# ----------------------------------------------------------------------
def knn_hypergraph(x, k, normalized=True):
    """
    Build hypergraph from KNN.
    Each node and its k nearest neighbors form a hyperedge.
    Returns:
        hyperedge_matrix: (B, N, k) – node indices in each hyperedge (padded with -1? no, all valid)
        point_hyperedge_index: (B, N, max_edges) – hyperedge indices for each node (padded with -1)
        centers: None (for compatibility)
    """
    B, C, N, _ = x.shape
    device = x.device

    # Prepare features
    x_norm = x.squeeze(-1).transpose(1, 2).contiguous()  # (B, N, C)
    if normalized:
        x_norm = F.normalize(x_norm, p=2, dim=-1)

    # Compute KNN indices
    dist = pairwise_distance(x_norm)  # (B, N, N)
    _, nn_idx = torch.topk(dist, k=k, dim=-1, largest=False)  # (B, N, k)
    hyperedge_matrix = nn_idx  # (B, N, k)

    # Build point-hyperedge mapping
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k).reshape(-1)
    node_idx = nn_idx.reshape(-1)
    hyperedge_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k).reshape(-1)

    keys = batch_idx * N + node_idx
    sorted_order = torch.argsort(keys)
    sorted_keys = keys[sorted_order]
    sorted_hyperedges = hyperedge_idx[sorted_order]

    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    max_edges = counts.max().item()

    # Scatter into point_hyperedge_index
    total_assignments = len(sorted_hyperedges)
    offsets = torch.zeros(len(counts), dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)[:-1]
    group_start = offsets.repeat_interleave(counts)
    inner_idx = torch.arange(total_assignments, device=device) - group_start

    b_vals = unique_keys // N
    p_vals = unique_keys % N
    b_expanded = b_vals.repeat_interleave(counts)
    p_expanded = p_vals.repeat_interleave(counts)

    linear_idx = b_expanded * (N * max_edges) + p_expanded * max_edges + inner_idx

    point_hyperedge_index = -torch.ones(B * N * max_edges, dtype=torch.long, device=device)
    point_hyperedge_index[linear_idx] = sorted_hyperedges
    point_hyperedge_index = point_hyperedge_index.reshape(B, N, max_edges)

    return hyperedge_matrix, point_hyperedge_index, None


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
        n_clusters = hyperedge_matrix.size(1)  # = N (each node is a hyperedge center)
        k = hyperedge_matrix.size(-1)
        device = x.device

        x_flat = x.reshape(B, C, N)  # (B, C, N)

        # --- Node -> hyperedge aggregation ---
        # hyperedge_matrix: (B, N, k) – each row i contains k neighbor indices (including i itself)
        # We need to collect features for each hyperedge (i.e., each node as a hyperedge)
        # We'll treat each hyperedge independently: (B, N, k) -> flatten to (B*N*k)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k).reshape(-1)  # (B*N*k)
        cluster_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k).reshape(-1)  # (B*N*k)
        node_idx = hyperedge_matrix.reshape(-1)  # (B*N*k)

        global_cluster = batch_idx * N + cluster_idx  # (B*N*k)
        global_node = batch_idx * N + node_idx        # (B*N*k)

        x_flat_2d = x_flat.permute(0, 2, 1).reshape(B * N, C).contiguous()  # (B*N, C)
        node_feats = x_flat_2d[global_node, :]  # (B*N*k, C)

        total_clusters = B * N
        hyperedge_sum = torch.zeros(total_clusters, C, device=device, dtype=x.dtype)
        hyperedge_sum.scatter_add_(0, global_cluster.unsqueeze(1).expand(-1, C), node_feats)

        hyperedge_cnt = torch.zeros(total_clusters, dtype=torch.long, device=device)
        ones = torch.ones_like(global_cluster)
        hyperedge_cnt.scatter_add_(0, global_cluster, ones)

        hyperedge_cnt = hyperedge_cnt.clamp(min=1)
        hyperedge_feat = hyperedge_sum / hyperedge_cnt.unsqueeze(1).float()  # (B*N, C)
        hyperedge_feat = hyperedge_feat.reshape(B, N, C).permute(0, 2, 1)    # (B, C, N)
        hyperedge_feat = hyperedge_feat.unsqueeze(-1)                         # (B, C, N, 1)

        # Transform hyperedge features
        hyperedge_feat = self.node_to_hyper(hyperedge_feat)                   # (B, C, N, 1)
        hyperedge_feat_sq = hyperedge_feat.squeeze(-1)                        # (B, C, N)

        # --- Hyperedge -> node aggregation ---
        # point_hyperedge_index: (B, N, max_edges) – for each node, list of hyperedge indices it belongs to
        max_edges = point_hyperedge_index.size(-1)
        mask_p = point_hyperedge_index != -1
        p_idx = point_hyperedge_index.clone()
        p_idx[~mask_p] = 0

        batch_idx_p = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, max_edges).reshape(-1)  # (B*N*max_edges)
        node_idx_p = torch.arange(N, device=device).view(1, N, 1).expand(B, N, max_edges).reshape(-1)   # (B*N*max_edges)
        hyperedge_idx_p = p_idx.reshape(-1)  # (B*N*max_edges)

        global_node_p = batch_idx_p * N + node_idx_p
        global_hyperedge_p = batch_idx_p * N + hyperedge_idx_p

        # Gather hyperedge features
        hyperedge_feat_2d = hyperedge_feat_sq.permute(0, 2, 1).reshape(total_clusters, C).contiguous()
        hyperedge_feat_for_nodes = hyperedge_feat_2d[global_hyperedge_p, :]  # (B*N*max_edges, C)

        total_nodes = B * N
        node_sum = torch.zeros(total_nodes, C, device=device, dtype=x.dtype)
        node_sum.scatter_add_(0, global_node_p.unsqueeze(1).expand(-1, C), hyperedge_feat_for_nodes)

        node_cnt = torch.zeros(total_nodes, dtype=torch.long, device=device)
        ones_p = torch.ones_like(global_node_p)
        node_cnt.scatter_add_(0, global_node_p, ones_p)

        node_cnt = node_cnt.clamp(min=1)
        node_aggr = node_sum / node_cnt.unsqueeze(1).float()  # (total_nodes, C)
        node_aggr = node_aggr.reshape(B, N, C).permute(0, 2, 1)  # (B, C, N)

        # GIN-style residual
        out = (1 + self.eps) * x_flat + node_aggr
        out = out.unsqueeze(-1)                                       # (B, C, N, 1)
        out = self.hyper_to_node(out)                                 # (B, out_channels, N, 1)
        return out.reshape(B, -1, H, W)


# ----------------------------------------------------------------------
# Grapher Block (with KNN hypergraph construction)
# ----------------------------------------------------------------------
class Grapher(nn.Module):
    def __init__(self, dim, k=9, dilation=1, conv='mr', act='gelu', norm='batch',
                 bias=True, stochastic=False, epsilon=0.0, reduce_ratio=1, n=196,
                 drop_path=0.0, relative_pos=False):
        super().__init__()
        self.k = k
        self.conv_type = conv
        if conv == 'hypergraph':
            # Hypergraph version
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
            # Original graph conv (not implemented here) – raise error or pass
            # For simplicity, we only support hypergraph in this version.
            raise NotImplementedError("Only hypergraph convolution is supported in this version.")

    @torch.no_grad()
    def _build_hypergraph(self, x_flat):
        return knn_hypergraph(x_flat, self.k)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H*W, 1)

        # Build hypergraph
        hyperedge_matrix, point_hyperedge_index, _ = self._build_hypergraph(x_flat)

        # Hypergraph convolution
        x = self.hypergraph_conv(x, hyperedge_matrix, point_hyperedge_index)
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
# Downsample (spatial reduction)
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
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        drop_path = opt.drop_path

        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]  # from original, may be unused
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # constant k

        # Stem and position embedding
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        trunc_normal_(self.pos_embed, std=0.02)

        HW = (224 // 4) * (224 // 4)  # 56*56 = 3136

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4  # spatial size reduces by 2 each side => area /4
            for j in range(blocks[i]):
                self.backbone += [
                    nn.Sequential(
                        Grapher(channels[i], k=num_knn[idx], dilation=min(idx//4+1, 49//num_knn[idx]),
                                conv=conv, act=act, norm=norm, bias=bias,
                                stochastic=stochastic, epsilon=epsilon,
                                reduce_ratio=reduce_ratios[i], n=HW,
                                drop_path=dpr[idx], relative_pos=True),
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
    'pvihg_knn_ti_224': _cfg(),
    'pvihg_knn_s_224': _cfg(),
    'pvihg_knn_m_224': _cfg(),
    'pvihg_knn_b_224': _cfg(crop_pct=0.95),
}


@register_model
def pvihg_knn_ti_224(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9                      # neighbor num
            self.conv = 'hypergraph'         # use hypergraph convolution
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]       # ti variant
            self.channels = [48, 96, 240, 384]
            self.n_classes = num_classes
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['pvihg_knn_ti_224']
    return model


@register_model
def pvihg_knn_s_224(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9
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
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['pvihg_knn_s_224']
    return model


@register_model
def pvihg_knn_m_224(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9
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
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['pvihg_knn_m_224']
    return model


@register_model
def pvihg_knn_b_224(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9
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
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['pvihg_knn_b_224']
    return model