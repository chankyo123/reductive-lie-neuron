"""
Reductive Lie Neurons (ReLNs) — quick start
============================================

A minimal, self-contained tour of the core library. It shows:

  1. The adjoint-invariant bilinear form  B(X, Y) = 2n*tr(XY) - tr(X)tr(Y)
     that powers ReLNs, and verifies numerically that it is invariant under
     the adjoint action of the *full* general linear group GL(n) -- i.e.
     B(gXg^-1, gYg^-1) = B(X, Y) for any invertible g.

  2. A small equivariant ReLN stack running a forward pass on sl(3) features.

Run from the repository root:

    python examples/quickstart.py

Requires: torch, einops, numpy  (see README for installation).
"""

import sys
from pathlib import Path

import torch

# The core library lives in relns/core and expects that
# directory on the path (its modules do `from core.lie_alg_util import *`).
LIB_ROOT = Path(__file__).resolve().parent.parent / "relns"
sys.path.insert(0, str(LIB_ROOT))

from core.lie_alg_util import HatLayer, vee, killingform, lie_bracket  # noqa: E402
from core.reln_layers import ReLNLinearAndKillingRelu             # noqa: E402


def demo_adjoint_invariant_form():
    """B(X, Y) = 2n*tr(XY) - tr(X)tr(Y) is invariant under conjugation by GL(n)."""
    print("=" * 64)
    print("1) Adjoint-invariant bilinear form on gl(3)")
    print("=" * 64)

    torch.manual_seed(0)
    hat = HatLayer("gl3")  # maps a 9-vector to a 3x3 matrix in gl(3)

    # Two random Lie-algebra elements, as 9-D coordinate vectors.
    X = torch.randn(9)
    Y = torch.randn(9)
    X_mat, Y_mat = hat(X), hat(Y)  # [3, 3] each

    B_xy = killingform(X_mat, Y_mat, algebra_type="gl3").item()

    # A random element g of GL(3) (rejection-sample until clearly invertible).
    g = torch.randn(3, 3)
    while g.det().abs() < 1e-2:
        g = torch.randn(3, 3)
    g_inv = torch.linalg.inv(g)

    # Adjoint action: conjugate both inputs by g.
    X_t = g @ X_mat @ g_inv
    Y_t = g @ Y_mat @ g_inv
    B_xy_t = killingform(X_t, Y_t, algebra_type="gl3").item()

    print(f"  B(X, Y)            = {B_xy: .6f}")
    print(f"  B(gXg^-1, gYg^-1)  = {B_xy_t: .6f}")
    print(f"  |difference|       = {abs(B_xy - B_xy_t): .3e}  (should be ~0)")
    assert abs(B_xy - B_xy_t) < 1e-3, "bilinear form is not adjoint-invariant!"
    print("  OK: the bilinear form is invariant under the GL(3) adjoint action.\n")


def demo_equivariant_forward():
    """Run a small ReLN stack on sl(3) features and report the output shape."""
    print("=" * 64)
    print("2) Equivariant ReLN forward pass on sl(3) features")
    print("=" * 64)

    # Feature convention: [batch, channels, algebra_dim, num_points].
    # sl(3) has dimension 8.
    B, F, K, N = 2, 4, 8, 16
    x = torch.randn(B, F, K, N)

    net = torch.nn.Sequential(
        ReLNLinearAndKillingRelu(F, 16, algebra_type="sl3", share_nonlinearity=False),
        ReLNLinearAndKillingRelu(16, 8, algebra_type="sl3", share_nonlinearity=False),
    )

    with torch.no_grad():
        y = net(x)

    print(f"  input  shape: {tuple(x.shape)}   [B, F, dim(sl3)=8, N]")
    print(f"  output shape: {tuple(y.shape)}")
    print("  OK: the ReLN layers are equivariant by construction.")
    print("  (See relns/experiment/sl3_inv_* for full equivariance tests.)\n")


if __name__ == "__main__":
    print(f"\nPyTorch {torch.__version__}\n")
    demo_adjoint_invariant_form()
    demo_equivariant_forward()
    print("Done. Next: see the README for reproducing the paper experiments.")
