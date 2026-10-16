# test_mc_case.py

import pytest
import numpy as np
from scipy.stats import norm
from monaco.mc_case import Case
from monaco.mc_var import InVar, OutVar


@pytest.fixture
def case():
    seed = 74494861
    invar = {
        "Test": InVar(
            "Test",
            ndraws=10,
            ninvar=1,
            dist=norm,
            distkwargs={"loc": 10, "scale": 4},
            seed=seed,
            firstcaseismedian=True,
        )
    }
    case = Case(ncase=0, ismedian=False, invars=invar, constvals=dict(), seed=seed)
    return case


def test_case_gen(case):
    assert case.invals["Test"].val == pytest.approx(10)
    assert case.vars["Test"].vals[0] == pytest.approx(10)
    assert case.vals["Test"].val == pytest.approx(10)


def test_case_getitem(case):
    assert case["Test"].val == pytest.approx(10)


def test_case_addoutvar(case):
    outvar = OutVar("TestOut", vals=list(range(10)))
    case.addOutVar(outvar)
    assert case.outvars["TestOut"].vals[0] == pytest.approx(0)
    assert case.vars["TestOut"].vals[0] == pytest.approx(0)


def test_case_addoutval(case):
    case.addOutVal("TestOut", [[0, 0], [0, 0], [0, 0]])
    assert case.outvals["TestOut"].val == [[0, 0], [0, 0], [0, 0]]
    assert case.outvals["TestOut"].shape == (3, 2)


def test_case_addoutval_with_valmap(case):
    valmap = {"a": 0, "b": -1, "c": -2, "d": -3, "e": -4, "f": -5}
    case.addOutVal("TestOut2", [["a", "b"], ["c", "d"], ["e", "f"]], valmap=valmap)
    np.testing.assert_array_equal(case.outvals["TestOut2"].num, [[0, -1], [-2, -3], [-4, -5]])


def test_case_get_out_vals(case):
    outvar = OutVar("TestOut", vals=["a"] * 5 + ["b"] * 5)
    case.addOutVar(outvar)
    assert case.getOutVals()["TestOut"].val == "a"


def test_case_lazy_invals(case):
    assert case._invals is None
    case.addOutVal("TestOut", [[1, 2], [3, 4]])
    assert list(case.vals.keys()) == ["Test", "TestOut", "TestOut [0]", "TestOut [1]"]
    case.addOutVal("TestOut2", 5)
    assert list(case.vals.keys())[-1] == "TestOut2"
    case.resetInVals()
    assert case._invals is None
    assert case["Test"].val == pytest.approx(10)


def test_case_pickle_drops_invals(case):
    import pickle
    from monaco.globals import register_global_vars

    case.addOutVal("TestOut", 1)
    assert case.invals["Test"].val == pytest.approx(10)
    state = case.__getstate__()
    assert "_invals" not in state and "_vals" not in state
    register_global_vars(case.invars, dict(), dict())
    try:
        case2 = pickle.loads(pickle.dumps(case))
    finally:
        register_global_vars(dict(), dict(), dict())
    assert case2._invals is None
    assert case2["Test"].val == pytest.approx(10)
    assert case2["TestOut"].val == 1
