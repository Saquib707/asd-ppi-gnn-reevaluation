"""
Edge-indexed (sparse) GAT layer -- mathematically identical to the dense
formulation in 03_models.py, but computes attention only over real edges.

Dense cost:  H x N x N   = 4 x 1245 x 1245 = 6,200,100 entries per layer
Sparse cost: H x |E'|    = 4 x    32,757   =   131,028 entries per layer
(|E'| = 2E + N self-loops)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_edge_index(A):
    """Dense adjacency -> edge index with self-loops added."""
    src, dst = torch.nonzero(torch.as_tensor(A) > 0, as_tuple=True)
    n = A.shape[0]
    loop = torch.arange(n)
    src = torch.cat([src, loop])
    dst = torch.cat([dst, loop])
    return torch.stack([src, dst])          # 2 x E'


def scatter_softmax(logits, index, n):
    """Softmax over incoming edges of each destination node. logits: H x E'."""
    h, e = logits.shape
    idx = index.unsqueeze(0).expand(h, e)
    mx = torch.full((h, n), float("-inf"), device=logits.device)
    mx = mx.scatter_reduce(1, idx, logits, reduce="amax", include_self=True)
    ex = torch.exp(logits - mx.gather(1, idx))
    den = torch.zeros((h, n), device=logits.device).scatter_add(1, idx, ex)
    return ex / (den.gather(1, idx) + 1e-16)


class SparseGATLayer(nn.Module):
    def __init__(self, fi, fo, heads=4, concat=True, dropout=0.2, alpha=0.01):
        super().__init__()
        self.heads, self.fo, self.concat = heads, fo, concat
        self.W = nn.Parameter(torch.empty(heads, fi, fo))
        self.a_src = nn.Parameter(torch.empty(heads, fo))
        self.a_dst = nn.Parameter(torch.empty(heads, fo))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        h = torch.einsum("nf,hfo->hno", x, self.W)            # H x N x F'
        es = (h * self.a_src[:, None, :]).sum(-1)             # H x N
        ed = (h * self.a_dst[:, None, :]).sum(-1)             # H x N
        # Message flows src -> dst. Following the dense convention, the attention
        # vector a_src is applied to the DESTINATION node and a_dst to the source;
        # the softmax normalises over the incoming edges of each destination.
        logits = self.leaky(es[:, dst] + ed[:, src])          # H x E'
        att = scatter_softmax(logits, dst, n)
        att = F.dropout(att, self.dropout, self.training)
        msg = h[:, src, :] * att.unsqueeze(-1)                # H x E' x F'
        out = torch.zeros(self.heads, n, self.fo, device=x.device, dtype=h.dtype)
        out.index_add_(1, dst, msg)
        self.last_attention = att.detach()
        self.last_edge_index = edge_index
        if self.concat:
            return out.permute(1, 0, 2).reshape(n, -1)
        return out.mean(0)


class SparseGAT(nn.Module):
    """Two sparse GAT layers + a linear (logistic-regression) output head."""

    def __init__(self, fin, hidden=8, heads=4, dropout=0.2, depth=2):
        super().__init__()
        self.dropout = dropout
        self.l1 = SparseGATLayer(fin, hidden, heads=heads, concat=True, dropout=dropout)
        self.l2 = SparseGATLayer(hidden * heads, 2, heads=1, concat=False, dropout=dropout)
        self.head = nn.Linear(2, 1)

    def forward(self, x, edge_index, return_emb=False):
        h = F.leaky_relu(self.l1(x, edge_index), 0.01)
        h = F.dropout(h, self.dropout, self.training)
        emb = F.leaky_relu(self.l2(h, edge_index), 0.01)
        logit = self.head(emb).squeeze(-1)
        return (logit, emb) if return_emb else logit
