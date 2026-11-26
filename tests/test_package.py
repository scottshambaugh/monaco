# test_package.py
import importlib
import subprocess
import sys

import pytest
import monaco as mc


def test_lazy_names_match_all():
    for modname, names in mc._LAZY_NAMES.items():
        assert list(names) == importlib.import_module(modname).__all__


def test_no_leaked_names():
    for name in ("np", "plt", "tqdm", "os", "warnings", "logging", "metadata", "cloudpickle"):
        assert not hasattr(mc, name)
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        mc.nope


def test_all_names_exist():
    assert len(mc.__all__) == len(set(mc.__all__))
    for name in mc.__all__:
        assert getattr(mc, name) is not None
    assert "GlobalsPlugin" not in mc.__all__


def test_star_import():
    namespace = {}
    exec("from monaco import *", namespace)
    assert "Sim" in namespace
    assert "plot_hist" in namespace
    assert "np" not in namespace


def test_import_does_not_load_optional_libraries():
    code = (
        "import sys, monaco\n"
        "monaco.Sim\n"
        "print(sorted(m for m in ('matplotlib', 'pandas', 'dask') if m in sys.modules))\n"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "[]"
