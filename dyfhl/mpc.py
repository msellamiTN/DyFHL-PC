"""Two-server additive secret sharing over F_p (p = 2^31 - 1), semi-honest, with a dealer for correlated
randomness (Beaver triples). Both servers run in this process; every value a server sends is counted, so
communication and rounds are those of the protocol. The pseudo-random generator is numpy's and is NOT
cryptographic: this is a prototype for correctness and cost, not a hardened implementation.

A shared value is a pair (x0, x1) of int64 arrays with x0 + x1 = x (mod p).
A masked input (class In) is a shared matrix X whose difference X - A to a dealer mask A has been opened
once; products of masked inputs then need no further communication (fixed-operand Beaver multiplication)."""
import numpy as np

P = (1 << 31) - 1
EB = 4                                   # bytes per field element


def mm(X, Y):
    """Matrix product mod P without int64 overflow (16-bit limbs of X)."""
    X, Y = X % P, Y % P
    hi, lo = X >> 16, X & 0xFFFF
    return (((hi @ Y) % P << 16) + (lo @ Y)) % P


class In:
    def __init__(self, sh, mask_sh, opened):
        self.sh, self.mask, self.d = sh, mask_sh, opened

    def cols(self, s, e):
        return In((self.sh[0][:, s:e], self.sh[1][:, s:e]), (self.mask[0][:, s:e], self.mask[1][:, s:e]),
                  self.d[:, s:e])


class MPC:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.online = 0          # bytes between the two servers
        self.inp = 0             # bytes from data owners to servers
        self.out = 0             # bytes from servers to the querier
        self.rounds = 0
        self.triples = 0         # scalar Beaver triples consumed
        self.mat_triples = 0     # scalar products inside matrix triples (dealer work)

    # ---- sharing
    def rnd(self, shape, nonzero=False):
        return self.rng.integers(1 if nonzero else 0, P, shape, dtype=np.int64)

    def _split(self, x):
        x0 = self.rnd(x.shape)
        return x0, (x - x0) % P

    def share(self, x):
        x = np.asarray(x, np.int64) % P
        self.inp += 2 * x.size * EB
        return self._split(x)

    def dealer_random(self, shape, nonzero=False):
        return self._split(self.rnd(shape, nonzero))

    def masked_input(self, x):
        """Owner shares x; servers open x - a for a dealer mask a (one message each way)."""
        sh = self.share(x)
        a = self._split(self.rnd(sh[0].shape))
        d = (sh[0] - a[0] + sh[1] - a[1]) % P
        self.online += 2 * d.size * EB
        return In(sh, a, d)

    # ---- linear operations (local)
    @staticmethod
    def add(x, y):
        return (x[0] + y[0]) % P, (x[1] + y[1]) % P

    @staticmethod
    def sub(x, y):
        return (x[0] - y[0]) % P, (x[1] - y[1]) % P

    @staticmethod
    def cadd(x, c):
        return (x[0] + c) % P, x[1]

    @staticmethod
    def neg(x):
        return (-x[0]) % P, (-x[1]) % P

    @staticmethod
    def scale(x, c):
        return x[0] * c % P, x[1] * c % P

    # ---- multiplications
    def mul(self, x, y):
        """Element-wise product of two shared arrays of equal shape: one triple per element, one round."""
        a, b = self.rnd(x[0].shape), self.rnd(x[0].shape)
        c = a * b % P
        a, b, c = self._split(a), self._split(b), self._split(c)
        d = (x[0] - a[0] + x[1] - a[1]) % P
        e = (y[0] - b[0] + y[1] - b[1]) % P
        self.online += 4 * d.size * EB
        self.rounds += 1
        self.triples += d.size
        z0 = (c[0] + d * b[0] % P + e * a[0] % P + d * e % P) % P
        z1 = (c[1] + d * b[1] % P + e * a[1] % P) % P
        return z0, z1

    def matmul(self, X: In, Y: In):
        """X @ Y^T for masked inputs: no communication; the dealer supplies shares of A_X @ A_Y^T."""
        ax, ay = (X.mask[0] + X.mask[1]) % P, (Y.mask[0] + Y.mask[1]) % P
        c = self._split(mm(ax, ay.T))
        self.mat_triples += ax.shape[0] * ay.shape[0] * ax.shape[1]
        dd = mm(X.d, Y.d.T)
        z0 = (c[0] + mm(X.d, Y.mask[0].T) + mm(X.mask[0], Y.d.T) + dd) % P
        z1 = (c[1] + mm(X.d, Y.mask[1].T) + mm(X.mask[1], Y.d.T)) % P
        return z0, z1

    def prod(self, x):
        """Product along the last axis by a balanced tree."""
        while x[0].shape[-1] > 1:
            m = x[0].shape[-1]
            h = m // 2
            l = (x[0][..., :h], x[1][..., :h])
            r = (x[0][..., h:2 * h], x[1][..., h:2 * h])
            y = self.mul(l, r)
            if m % 2:
                y = (np.concatenate([y[0], x[0][..., -1:]], -1), np.concatenate([y[1], x[1][..., -1:]], -1))
            x = y
        return x[0][..., 0], x[1][..., 0]

    def reveal(self, x):
        """Both servers send their share to the querier."""
        self.out += 2 * x[0].size * EB
        return (x[0] + x[1]) % P

    def cost(self):
        return dict(online_MB=self.online / 1e6, input_MB=self.inp / 1e6, output_MB=self.out / 1e6,
                    rounds=self.rounds, triples=self.triples, mat_triple_products=self.mat_triples)


# --------------------------------------------------------------------------- exact cross-domain checks
def expand(H, Wm, slices):
    """Incidence rows with a wildcard written as the full value set of its attribute."""
    X = H.astype(np.int64).copy()
    for a, (s, e) in enumerate(slices):
        X[Wm[:, a].astype(bool), s:e] = 1
    return X


def exact_checks(mpc: MPC, XI, effI, XJ, effJ, slices, Q=None):
    """Flags for the querier's rules I against the foreign rules J of one scope; nothing else is revealed.
    XI [n,V], XJ [N,V]: expanded incidence rows (each owner shares its own rows).
    conflict_i  = exists j: effects differ and the value sets intersect on every attribute
    redundant_i = exists j: same effect and rule i is contained in rule j
    gap_q       = no j matches request q."""
    n, N = len(XI), len(XJ)
    I, J = mpc.masked_input(XI), mpc.masked_input(XJ)
    eI, eJ = mpc.masked_input(effI[:, None]), mpc.masked_input(effJ[:, None])
    ee = mpc.matmul(eI, eJ)                                               # e_i e_j
    xor = mpc.sub(mpc.add((np.repeat(eI.sh[0], N, 1), np.repeat(eI.sh[1], N, 1)),
                          (np.repeat(eJ.sh[0].T, n, 0), np.repeat(eJ.sh[1].T, n, 0))), mpc.scale(ee, 2))
    # conflict: product over attributes of the per-attribute inner products, times xor, random linear sum
    acc = xor
    for s, e in slices:
        acc = mpc.mul(acc, mpc.matmul(I.cols(s, e), J.cols(s, e)))
    z = mpc.mul(acc, mpc.dealer_random((n, N)))
    conflict = mpc.reveal((z[0].sum(1) % P, z[1].sum(1) % P)) != 0
    # redundancy: |i \ j| + xor is zero iff contained and same effect; product over j is zero iff some j
    size = (I.sh[0].sum(1, keepdims=True) % P, I.sh[1].sum(1, keepdims=True) % P)
    diff = mpc.sub((np.repeat(size[0], N, 1), np.repeat(size[1], N, 1)), mpc.matmul(I, J))
    pr = mpc.prod(mpc.add(diff, xor))
    redundant = mpc.reveal(mpc.mul(pr, mpc.dealer_random((n,), nonzero=True))) == 0
    gap = None
    if Q is not None and len(Q):
        Qi = mpc.masked_input(Q)
        miss = mpc.cadd(mpc.neg(mpc.matmul(Qi, J)), len(slices))         # A - <q, j>: zero iff j matches q
        pq = mpc.prod(miss)
        gap = mpc.reveal(mpc.mul(pq, mpc.dealer_random((len(Q),), nonzero=True))) != 0
    return conflict, redundant, gap
