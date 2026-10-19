"""
Fast Hypergraph Vision Graph Neural Network (ViHG)
Based on original ViG implementation, with KNN-based hypergraph construction.
All loops vectorized for maximum speed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model

# -------------------- Helper functions --------------------
def pairwise_distance(x):
    """
    Compute pairwise distance for a batch of point sets.
    x: (B, N, C)
    return: (B, N, N) distance matrix
    """
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(x ** 2, dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

def batched_index_select(x, idx, pad_value=0):
    """
    Batched index select with -1 padding.
    Args:
        x: (B, C, N, 1)
        idx: (B, N, k) with possible -1
    Returns:
        (B, C, N, k) where invalid indices are filled with pad_value
    """
    B, C, N, _ = x.shape
    device = x.device
    mask = idx == -1
    idx = idx.clone()
    idx[mask] = 0  # temporarily replace for gather

    idx_base = torch.arange(B, device=device).view(B, 1, 1) * N
    idx_flat = (idx + idx_base).reshape(-1)

    x_flat = x.squeeze(-1).transpose(1, 2).contiguous().reshape(B * N, C)  # (B*N, C)
    selected = x_flat[idx_flat, :].reshape(B, N, -1, C).permute(0, 3, 1, 2)  # (B, C, N, k)

    # Zero out invalid positions
    selected = selected.masked_fill(mask.unsqueeze(1).expand_as(selected), pad_value)
    return selected

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

# -------------------- Hypergraph Construction (Fully Vectorized) --------------------
def knn_hypergraph(x, k, normalized=True):
    """
    Construct hypergraph using KNN.
    Each node and its k nearest neighbors form a hyperedge.
    Args:
        x: (B, C, N, 1) feature map flattened
        k: number of neighbors (including self)
        normalized: whether to L2 normalize features
    Returns:
        hyperedge_matrix: (B, N, k) indices of nodes in each hyperedge
        point_hyperedge_index: (B, N, max_edges) for each node, which hyperedges it belongs to (padded with -1)
        centers: None (for compatibility)
    """
    B, C, N, _ = x.shape
    device = x.device

    # Prepare features for distance computation
    x_norm = x.squeeze(-1).transpose(1, 2).contiguous()  # (B, N, C)
    if normalized:
        x_norm = F.normalize(x_norm, p=2, dim=-1)

    # Compute pairwise distance and get k nearest neighbors
    dist = pairwise_distance(x_norm)  # (B, N, N)
    _, nn_idx = torch.topk(dist, k=k, dim=-1, largest=False)  # (B, N, k)

    hyperedge_matrix = nn_idx  # (B, N, k)

    # Generate all triplets (batch, node, hyperedge)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k).reshape(-1)
    node_idx = nn_idx.reshape(-1)  # (B*N*k)
    hyperedge_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k).reshape(-1)

    # Group by (batch, node)
    keys = batch_idx * N + node_idx
    sorted_order = torch.argsort(keys)
    sorted_keys = keys[sorted_order]
    sorted_hyperedges = hyperedge_idx[sorted_order]

    # Get counts per group
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    max_edges = counts.max().item()

    total_assignments = len(sorted_hyperedges)

    # Compute start index of each group (offsets)
    offsets = torch.zeros(len(counts), dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)[:-1]  # start indices
    # Expand offsets to per-assignment
    group_start = offsets.repeat_interleave(counts)
    # Inner index within group = global index - group_start
    inner_idx = torch.arange(total_assignments, device=device) - group_start

    # Expand b and p to per-assignment
    b_vals = unique_keys // N
    p_vals = unique_keys % N
    b_expanded = b_vals.repeat_interleave(counts)
    p_expanded = p_vals.repeat_interleave(counts)

    # Linear index into flattened point_hyperedge_index
    linear_idx = b_expanded * (N * max_edges) + p_expanded * max_edges + inner_idx

    # Initialize and scatter
    point_hyperedge_index = -torch.ones(B * N * max_edges, dtype=torch.long, device=device)
    point_hyperedge_index[linear_idx] = sorted_hyperedges
    point_hyperedge_index = point_hyperedge_index.reshape(B, N, max_edges)

    return hyperedge_matrix, point_hyperedge_index, None

# -------------------- Hypergraph Convolution --------------------
class HypergraphConv2d(nn.Module):
    """
    Hypergraph convolution based on GIN-style message passing.
    """
    def __init__(self, in_channels, out_channels, act='gelu', norm='batch', bias=True):
        super().__init__()
        # Node -> hyperedge transformation
        self.node_to_hyper = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=bias),
            norm_layer(norm, in_channels),
            act_layer(act)
        )
        # Hyperedge -> node transformation
        self.hyper_to_node = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            norm_layer(norm, out_channels)
        )
        self.eps = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, hyperedge_matrix, point_hyperedge_index):
        """
        x: (B, C, H, W)
        hyperedge_matrix: (B, N, k) node indices in each hyperedge
        point_hyperedge_index: (B, N, max_edges) hyperedge indices for each node
        """
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.reshape(B, C, N, 1)  # (B, C, N, 1)

        # Node to hyperedge aggregation (mean pooling)
        hyperedge_nodes = batched_index_select(x_flat, hyperedge_matrix)  # (B, C, N, k)
        hyperedge_feat = hyperedge_nodes.mean(dim=-1, keepdim=True)  # (B, C, N, 1)

        # Transform hyperedge features
        hyperedge_feat = self.node_to_hyper(hyperedge_feat)  # (B, C, N, 1)

        # Hyperedge to node aggregation
        node_from_hyper = batched_index_select(hyperedge_feat, point_hyperedge_index)  # (B, C, N, max_edges)
        node_aggr = node_from_hyper.mean(dim=-1, keepdim=True)  # (B, C, N, 1)

        # Residual connection with learnable epsilon (like GIN)
        out = (1 + self.eps) * x_flat + node_aggr
        out = self.hyper_to_node(out)  # (B, out_channels, N, 1)
        return out.reshape(B, -1, H, W)

# -------------------- Grapher Module --------------------
class Grapher(nn.Module):
    """
    Grapher block with hypergraph convolution and FFN.
    """
    def __init__(self, dim, k=9, act='gelu', norm='batch', bias=True,
                 drop_path=0.0, r=1, n=196):
        super().__init__()
        self.dim = dim
        self.k = k
        self.n = n
        self.r = r

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

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.reshape(B, C, N, 1)

        # Build hypergraph
        hyperedge_matrix, point_hyperedge_index, _ = knn_hypergraph(x_flat, self.k)

        # Hypergraph convolution
        x = self.hypergraph_conv(x, hyperedge_matrix, point_hyperedge_index)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

# -------------------- Feed-Forward Network --------------------
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, act='gelu', norm='batch', drop_path=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            norm_layer(norm, hidden_dim)
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            norm_layer(norm, dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

# -------------------- Stem (Patch Embedding) --------------------
class Stem(nn.Module):
    """Overlapping patch embedding (4 downsampling stages)"""
    def __init__(self, in_chans=3, out_dim=192, act='gelu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_chans, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )
        # Check divisibility
        assert out_dim % 8 == 0, "out_dim must be multiple of 8"

    def forward(self, x):
        return self.convs(x)

# -------------------- Main Model --------------------
class DeepGCN(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, drop_path_rate=0.0, dropout=0.0,
                 k=9, blocks=12, channels=192, act='gelu', norm='batch', bias=True,
                 use_dilation=True):
        super().__init__()
        self.num_blocks = blocks
        self.channels = channels
        self.k = k

        # Stem
        self.stem = Stem(in_chans, channels, act)

        # Position embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        trunc_normal_(self.pos_embed, std=0.02)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, blocks)]

        # Backbone blocks
        self.blocks = nn.ModuleList()
        for i in range(blocks):
            # Gradually increase k (dilation effect)
            if use_dilation:
                k_block = min(k + i, 2*k)  # linear increase up to 2k
            else:
                k_block = k
            self.blocks.append(nn.Sequential(
                Grapher(channels, k=k_block, act=act, norm=norm, bias=bias,
                        drop_path=dpr[i], n=196),
                FFN(channels, channels*4, act=act, norm=norm, drop_path=dpr[i])
            ))

        # Prediction head
        self.head = nn.Sequential(
            nn.Conv2d(channels, 1024, 1, bias=True),
            norm_layer(norm, 1024),
            act_layer(act),
            nn.Dropout(dropout),
            nn.Conv2d(1024, num_classes, 1, bias=True)
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

    def forward(self, x):
        x = self.stem(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.head(x)
        return x.squeeze(-1).squeeze(-1)

# -------------------- Model Variants --------------------
default_cfgs = {
    'vihg_ti_224': {
        'url': '',
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'first_conv': 'stem.convs.0',
        'classifier': 'head.4',
    }
}

@register_model
def vihgknnf_ti_224(pretrained=False, **kwargs):
    model = DeepGCN(
        in_chans=3,
        num_classes=kwargs.get('num_classes', 1000),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        dropout=kwargs.get('drop_rate', 0.0),
        k=9,           # initial neighbor number
        blocks=12,
        channels=192,
        act='gelu',
        norm='batch',
        bias=True,
        use_dilation=True,
    )
    model.default_cfg = default_cfgs['vihg_ti_224']
    if pretrained:
        # Load pretrained weights if available
        pass
    return model

@register_model
def vihgknnf_s_224(pretrained=False, **kwargs):
    model = DeepGCN(
        in_chans=3,
        num_classes=kwargs.get('num_classes', 1000),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        dropout=kwargs.get('drop_rate', 0.0),
        k=9,
        blocks=16,
        channels=320,
        act='gelu',
        norm='batch',
        bias=True,
        use_dilation=True,
    )
    model.default_cfg = default_cfgs['vihg_ti_224']
    return model

@register_model
def vihgknnf_b_224(pretrained=False, **kwargs):
    model = DeepGCN(
        in_chans=3,
        num_classes=kwargs.get('num_classes', 1000),
        drop_path_rate=kwargs.get('drop_path_rate', 0.0),
        dropout=kwargs.get('drop_rate', 0.0),
        k=9,
        blocks=16,
        channels=640,
        act='gelu',
        norm='batch',
        bias=True,
        use_dilation=True,
    )
    model.default_cfg = default_cfgs['vihg_ti_224']
    return model