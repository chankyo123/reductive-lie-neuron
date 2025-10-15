import torch
from torch import nn

def build_X_from_p(p, eta_diag):
    """
    Build the 5x5 embedded Lie algebra matrix X(p) for a batch of 4-vectors p.
    """
    num_edges = p.shape[0]
    X = p.new_zeros((num_edges, 5, 5))
    X[:, :4, 4] = p
    X[:, 4, :4] = p * eta_diag.unsqueeze(0)
    return X

def killingform_gl5(x_hat, d_hat):
    """
    Compute killing form for gl(5) using the full, general formula.
    """
    # tr(XY)
    tr_xy = (x_hat.transpose(-1, -2) * d_hat).sum(dim=(-1, -2))
    # tr(X) and tr(Y)
    tr_x = x_hat.diagonal(dim1=-2, dim2=-1).sum(-1)
    tr_y = d_hat.diagonal(dim1=-2, dim2=-1).sum(-1)
    
    # k(X,Y) = 2n*tr(XY) - 2*tr(X)*tr(Y), with n=5
    k = 10.0 * tr_xy - 2.0 * tr_x * tr_y
    return k.unsqueeze(-1)

class LGEB(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout = 0., c_weight=1.0, last_layer=False):
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2 # dims for Minkowski norm & inner product

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU())

        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output))

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi_x = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            layer)

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid())
        
        self.last_layer = last_layer
        if last_layer:
            del self.phi_x
            
        self.register_buffer('eta_diag', torch.tensor([1.0, -1.0, -1.0, -1.0]))

    def m_model(self, hi, hj, norms, dots):
        out = torch.cat([hi, hj, norms, dots], dim=1)
        out = self.phi_e(out)
        w = self.phi_m(out)
        out = out * w
        return out

    def h_model(self, h, edges, m, node_attr):
        i, j = edges
        agg = unsorted_segment_sum(m, i, num_segments=h.size(0))
        agg = torch.cat([h, agg, node_attr], dim=1)
        out = h + self.phi_h(agg)
        return out

    def x_model(self, x, edges, x_diff, m):
        i, j = edges
        trans = x_diff * self.phi_x(m)
        # From https://github.com/vgsatorras/egnn
        # This is never activated but just in case it explosed it may save the train
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, i, num_segments=x.size(0))
        x = x + agg * self.c_weight
        return x

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = normsq4(x_diff).unsqueeze(1)
        dots = dotsq4(x[i], x[j]).unsqueeze(1)
        norms, dots = psi(norms), psi(dots)
        return norms, dots, x_diff
    
    def minkowski_feats_kf(self, edges, x):
        i, j = edges
        p, q = x[i], x[j]
        x_diff = p - q

        # 1. Lie Algebra Killing Form (Invariant)
        # Build embedded Lie matrices X(p), X(q) as 5x5
        Xp = build_X_from_p(p, self.eta_diag)
        Xq = build_X_from_p(q, self.eta_diag)
        killing_form_norm = killingform_gl5(Xp, Xp)
        killing_form_dot = killingform_gl5(Xp, Xq)
        norms, dots = psi(killing_form_norm), psi(killing_form_dot)
        return norms, dots, x_diff
    
    def forward(self, h, x, edges, node_attr=None):
        i, j = edges
        norms, dots, x_diff = self.minkowski_feats(edges, x)
        # norms, dots, x_diff = self.minkowski_feats_kf(edges, x)

        m = self.m_model(h[i], h[j], norms, dots) # [B*N, hidden]
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m

class LorentzNet(nn.Module):
    r''' Implimentation of LorentzNet.

    Args:
        - `n_scalar` (int): number of input scalars.
        - `n_hidden` (int): dimension of latent space.
        - `n_class`  (int): number of output classes.
        - `n_layers` (int): number of LGEB layers.
        - `c_weight` (float): weight c in the x_model.
        - `dropout`  (float): dropout rate.
    '''
    def __init__(self, n_scalar, n_hidden, n_class = 2, n_layers = 6, c_weight = 1e-3, dropout = 0.):
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList([LGEB(self.n_hidden, self.n_hidden, self.n_hidden, 
                                    n_node_attr=n_scalar, dropout=dropout,
                                    c_weight=c_weight, last_layer=(i==n_layers-1))
                                    for i in range(n_layers)])
        self.graph_dec = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(self.n_hidden, n_class)) # classification

    def forward(self, scalars, x, edges, node_mask, edge_mask, n_nodes):
        
        # print("--- LorentzNet Forward Pass 시작 ---")
        # print(f"  [초기 입력] scalars shape: {scalars.shape}")
        # print(f"  [초기 입력] x (pmu) shape: {x.shape}")
        # print(f"  [초기 입력] edges[0] (i) shape: {edges[0].shape}")
        # print("-" * 35)
        
        h = self.embedding(scalars)
        # print(f"  [임베딩 후] h (스칼라 특징) shape: {h.shape}\n")
        for i in range(self.n_layers):
            h, x, _ = self.LGEBs[i](h, x, edges, node_attr=scalars)
            # print(f"  [LGEB {i+1} 후] h (업데이트된 스칼라) shape: {h.shape}")
            # print(f"  [LGEB {i+1} 후] x (업데이트된 좌표) shape: {x.shape}\n")

        h = h * node_mask
        h = h.view(-1, n_nodes, self.n_hidden)
        # print(f"  [Reshape 후] h shape: {h.shape}")
        h = torch.mean(h, dim=1)
        # print(f"  [평균 풀링 후] h (그래프 특징) shape: {h.shape}")
        pred = self.graph_dec(h)
        # print(f"  [최종 예측] pred shape: {pred.shape}")
        # print("-" * 35)
        return pred.squeeze(1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def normsq4(p):
    r''' Minkowski square norm
         `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    ''' 
    psq = torch.pow(p, 2)
    # print("=== normsq4 디버그 출력 ===")
    # print(psq.shape)
    # print((2 * psq[..., 0] - psq.sum(dim=-1)).shape)
    return 2 * psq[..., 0] - psq.sum(dim=-1)
    
def dotsq4(p,q):
    r''' Minkowski inner product
         `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    '''
    psq = p*q
    # print("=== dotsq4 디버그 출력 ===")
    # print(psq.shape)
    # print((2 * psq[..., 0] - psq.sum(dim=-1)).shape)
    return 2 * psq[..., 0] - psq.sum(dim=-1)
    
def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    '''
    return torch.sign(p) * torch.log(torch.abs(p) + 1)
