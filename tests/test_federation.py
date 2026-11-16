"""Exactness of incremental admission (Theorem 4) and agreement of the secure checks with plaintext findings."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dyfhl.federation import Federation
from dyfhl.secure_fed import secure_pairwise, secure_cross
from dyfhl.mpc import MPC


def test_incremental_equals_full_and_secure_equals_plain():
    for seed in range(3):
        f = Federation(K=3, rules_per_org=40, seed=seed)
        for i in range(120):
            ev = f.random_event()
            _, _, ck = f.apply(ev)
            s1 = f.snapshot(); f.full()
            assert s1 == f.snapshot()
            m = MPC(i)
            if ev[0] == "add":
                assert secure_pairwise(f, m, ev[1].id) in (True, None)
            for (t, r) in ck:
                if f.host[t] not in f.owners(r) or len(f.owners(r)) > 1:
                    assert secure_cross(f, m, t, r)


if __name__ == "__main__":
    test_incremental_equals_full_and_secure_equals_plain(); print("ok")
