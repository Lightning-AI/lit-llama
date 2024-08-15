# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import tempfile
import pathlib

import torch


class ATensor(torch.Tensor):
    pass


def test_lazy_load_basic(lit_llama):
    import lit_llama.utils

    with tempfile.TemporaryDirectory() as tmpdirname:
        m = torch.nn.Linear(5, 3)
        path = pathlib.Path(tmpdirname)
        fn = str(path / "test.pt")
        torch.save(m.state_dict(), fn)
        with lit_llama.utils.lazy_load(fn) as sd_lazy:
            assert "NotYetLoadedTensor" in str(next(iter(sd_lazy.values())))
            m2 = torch.nn.Linear(5, 3)
            m2.load_state_dict(sd_lazy)

        x = torch.randn(2, 5)
        actual = m2(x)
        expected = m(x)
        torch.testing.assert_close(actual, expected)


def test_lazy_load_subclass(lit_llama):
    import lit_llama.utils

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = pathlib.Path(tmpdirname)
        fn = str(path / "test.pt")
        t = torch.randn(2, 3)[:, 1:]
        sd = {
            1: t,
            2: torch.nn.Parameter(t),
            3: torch.Tensor._make_subclass(ATensor, t),
        }
        torch.save(sd, fn)
        with lit_llama.utils.lazy_load(fn) as sd_lazy:
            for k in sd.keys():
                actual = sd_lazy[k]
                expected = sd[k]
                torch.testing.assert_close(actual._load_tensor(), expected)


def test_incremental_write(tmp_path, lit_llama):
    import lit_llama.utils

    sd = {str(k): torch.randn(5, 10) for k in range(3)}
    sd_expected = {k: v.clone() for k, v in sd.items()}
    fn = str(tmp_path / "test.pt")
    with lit_llama.utils.incremental_save(fn) as f:
        sd["0"] = f.store_early(sd["0"])
        sd["2"] = f.store_early(sd["2"])
        f.save(sd)
    sd_actual = torch.load(fn)
    assert sd_actual.keys() == sd_expected.keys()
    for k, v_expected in sd_expected.items():
        v_actual = sd_actual[k]
        torch.testing.assert_close(v_expected, v_actual)


def test_find_multiple(lit_llama):
    from lit_llama.utils import find_multiple

    assert find_multiple(17, 5) == 20
    assert find_multiple(30, 7) == 35
    assert find_multiple(10, 2) == 10
    assert find_multiple(5, 10) == 10
