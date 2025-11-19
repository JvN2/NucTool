import sys
from pathlib import Path

# Add nuctool directory to path
nuctool_path = Path(__file__).parent.parent / "nuctool"
sys.path.insert(0, str(nuctool_path))

import numpy as np
from ChromatinFibers import ChromatinFiber


def test_occupancy_basic():
    seq = "A" * 1000
    cf = ChromatinFiber(seq)
    cf.place_dyads([100, 300, 700])
    occ = cf.occupancy(footprint=146)
    assert occ.shape[0] == len(seq)
    # check that dyad centers are within occupied windows
    for d in cf.dyads:
        assert occ[d] == 1.0


def test_sample_configuration():
    seq = "A" * 500
    cf = ChromatinFiber(seq)
    cf.place_dyads([50, 150, 250, 350])
    res = cf.sample_configuration(footprint=146, rng=np.random.default_rng(42))
    assert isinstance(res.dyads, np.ndarray)
    assert isinstance(res.occupancy, np.ndarray)
    assert res.occupancy.shape[0] == len(seq)
