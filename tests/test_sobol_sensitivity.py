# test_sobol_sensitivity.py
"""Tests for Sobol' sensitivity analysis functionality."""

import pytest
import numpy as np
from scipy.stats import uniform

import monaco as mc
from monaco.mc_sampling import saltelli_sampling, SaltelliSamples, get_saltelli_total_cases
from monaco.sobol_sensitivity import (
    calc_sobol_indices_from_saltelli,
    SobolIndices,
    bootstrap_sobol_indices,
)
from monaco.mc_enums import SampleMethod


# --- Saltelli Sampling Tests ---


class TestSaltelliSampling:
    """Tests for the saltelli_sampling function."""

    def test_sample_dimensions(self):
        """Test that sample matrices have correct dimensions."""
        nstars, ninvars, npts = 64, 3, 1
        samples = saltelli_sampling(nstars, ninvars, npts, seed=42)

        assert samples.A.shape == (nstars, ninvars)
        assert samples.B.shape == (nstars, ninvars)
        assert samples.AB.shape == (ninvars, npts, nstars, ninvars)

    def test_total_points_npts_1(self):
        """Test total points calculation with npts=1 (standard Saltelli)."""
        nstars, ninvars, npts = 64, 3, 1
        samples = saltelli_sampling(nstars, ninvars, npts, seed=42)

        expected_total = nstars * (2 + ninvars * npts)  # 64 * (2 + 3) = 320
        assert samples.total_points == expected_total
        assert get_saltelli_total_cases(nstars, ninvars, npts) == expected_total

    def test_total_points_npts_5(self):
        """Test total points calculation with npts=5."""
        nstars, ninvars, npts = 32, 4, 5
        samples = saltelli_sampling(nstars, ninvars, npts, seed=42)

        expected_total = nstars * (2 + ninvars * npts)  # 32 * (2 + 20) = 704
        assert samples.total_points == expected_total

    def test_values_in_unit_hypercube(self):
        """All sample values must be in [0, 1]."""
        samples = saltelli_sampling(128, 5, 2, seed=42)
        all_points = samples.get_all_points()

        assert np.all(all_points >= 0)
        assert np.all(all_points <= 1)

    def test_ab_interpolation_npts_1(self):
        """With npts=1, AB[i,0] should have column i equal to B[:,i]."""
        nstars, ninvars = 32, 3
        samples = saltelli_sampling(nstars, ninvars, npts=1, seed=42)

        for i in range(ninvars):
            # Column i of AB[i,0] should equal column i of B
            np.testing.assert_array_almost_equal(
                samples.AB[i, 0, :, i], samples.B[:, i]
            )
            # Other columns should equal A
            for j in range(ninvars):
                if j != i:
                    np.testing.assert_array_almost_equal(
                        samples.AB[i, 0, :, j], samples.A[:, j]
                    )

    def test_ab_interpolation_npts_2(self):
        """With npts=2, test interpolation at midpoint and endpoint."""
        nstars, ninvars = 32, 2
        samples = saltelli_sampling(nstars, ninvars, npts=2, seed=42)

        for i in range(ninvars):
            # At k=0 (t=0.5), column i should be midpoint between A and B
            midpoint = 0.5 * samples.A[:, i] + 0.5 * samples.B[:, i]
            np.testing.assert_array_almost_equal(
                samples.AB[i, 0, :, i], midpoint
            )

            # At k=1 (t=1.0), column i should equal B
            np.testing.assert_array_almost_equal(
                samples.AB[i, 1, :, i], samples.B[:, i]
            )

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical results."""
        s1 = saltelli_sampling(64, 3, 1, seed=42)
        s2 = saltelli_sampling(64, 3, 1, seed=42)

        np.testing.assert_array_equal(s1.A, s2.A)
        np.testing.assert_array_equal(s1.B, s2.B)
        np.testing.assert_array_equal(s1.AB, s2.AB)

    def test_different_with_different_seed(self):
        """Different seeds should produce different results."""
        s1 = saltelli_sampling(64, 3, 1, seed=42)
        s2 = saltelli_sampling(64, 3, 1, seed=43)

        assert not np.allclose(s1.A, s2.A)

    def test_get_all_points_shape(self):
        """Test get_all_points returns correct shape."""
        nstars, ninvars, npts = 32, 4, 2
        samples = saltelli_sampling(nstars, ninvars, npts, seed=42)

        all_points = samples.get_all_points()
        assert all_points.shape == (samples.total_points, ninvars)

    def test_get_point_labels_length(self):
        """Test get_point_labels returns correct number of labels."""
        nstars, ninvars, npts = 32, 3, 2
        samples = saltelli_sampling(nstars, ninvars, npts, seed=42)

        labels = samples.get_point_labels()
        assert len(labels) == samples.total_points


# --- Sobol Indices Calculation Tests ---


class TestSobolIndicesCalculation:
    """Tests for Sobol' sensitivity index calculations."""

    def test_linear_model_first_order(self):
        """Test with a linear model where sensitivity is proportional to coefficient squared."""
        # y = 3*x1 + 1*x2
        # V(y) = 9*V(x1) + 1*V(x2) = 9/12 + 1/12 = 10/12
        # S1 = 9/12 / (10/12) = 0.9
        # S2 = 1/12 / (10/12) = 0.1

        nstars = 2048
        samples = saltelli_sampling(nstars, ninvars=2, npts=1, seed=42)

        # Evaluate linear model at all points
        def linear_model(x):
            return 3 * x[:, 0] + 1 * x[:, 1]

        f_A = linear_model(samples.A)
        f_B = linear_model(samples.B)
        f_AB = np.zeros((2, nstars))
        f_AB[0] = linear_model(samples.AB[0, 0])
        f_AB[1] = linear_model(samples.AB[1, 0])

        indices = calc_sobol_indices_from_saltelli(
            f_A=f_A,
            f_B=f_B,
            f_AB=f_AB,
            invar_names=["x1", "x2"],
        )

        # Check first-order indices (with tolerance for Monte Carlo error)
        assert indices.first_order["x1"] == pytest.approx(0.9, abs=0.05)
        assert indices.first_order["x2"] == pytest.approx(0.1, abs=0.05)

        # For linear model, total order should equal first order
        assert indices.total_order["x1"] == pytest.approx(0.9, abs=0.05)
        assert indices.total_order["x2"] == pytest.approx(0.1, abs=0.05)

    def test_constant_output_zero_indices(self):
        """When output has no variance, all indices should be zero."""
        nstars = 64
        samples = saltelli_sampling(nstars, ninvars=2, npts=1, seed=42)

        # Constant output
        f_A = np.ones(nstars) * 5.0
        f_B = np.ones(nstars) * 5.0
        f_AB = np.ones((2, nstars)) * 5.0

        indices = calc_sobol_indices_from_saltelli(
            f_A=f_A, f_B=f_B, f_AB=f_AB, invar_names=["x1", "x2"]
        )

        assert indices.first_order["x1"] == 0.0
        assert indices.first_order["x2"] == 0.0

    def test_different_methods(self):
        """Test that different estimator methods work."""
        nstars = 512
        samples = saltelli_sampling(nstars, ninvars=2, npts=1, seed=42)

        def model(x):
            return x[:, 0] ** 2 + x[:, 1]

        f_A = model(samples.A)
        f_B = model(samples.B)
        f_AB = np.array([model(samples.AB[i, 0]) for i in range(2)])

        for method in ["saltelli_2002", "saltelli_2010", "jansen"]:
            indices = calc_sobol_indices_from_saltelli(
                f_A=f_A, f_B=f_B, f_AB=f_AB,
                invar_names=["x1", "x2"],
                method=method,
            )
            # All methods should produce reasonable indices
            assert -0.2 <= indices.first_order["x1"] <= 1.0
            assert -0.2 <= indices.first_order["x2"] <= 1.0

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            calc_sobol_indices_from_saltelli(
                f_A=np.zeros(10),
                f_B=np.zeros(10),
                f_AB=np.zeros((2, 10)),
                invar_names=["x1", "x2"],
                method="invalid_method",
            )


# --- Integration Tests with Sim ---


def ishigami_preprocess(case):
    """Preprocess for Ishigami function."""
    x1 = case.invals["x1"].val
    x2 = case.invals["x2"].val
    x3 = case.invals["x3"].val
    return (x1, x2, x3)


def ishigami_run(x1, x2, x3):
    """Ishigami function - standard sensitivity analysis benchmark."""
    a = 7.0
    b = 0.1
    y = np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1)
    return (y,)


def ishigami_postprocess(case, y):
    """Postprocess for Ishigami function."""
    case.addOutVal("y", y)


# Analytical Sobol' indices for Ishigami function with a=7, b=0.1
# Reference: Saltelli et al. (2008)
ISHIGAMI_S1 = {"x1": 0.3139, "x2": 0.4424, "x3": 0.0}
ISHIGAMI_ST = {"x1": 0.5576, "x2": 0.4424, "x3": 0.2437}


class TestSobolSimIntegration:
    """Integration tests with monaco Sim class."""

    @pytest.fixture
    def ishigami_sim(self):
        """Create an Ishigami simulation with Saltelli sampling."""
        fcns = {
            "preprocess": ishigami_preprocess,
            "run": ishigami_run,
            "postprocess": ishigami_postprocess,
        }

        # Use nstars=512 for reasonable accuracy
        sim = mc.Sim(
            name="ishigami_sobol_test",
            ndraws=512,  # Will be adjusted by Saltelli sampling
            fcns=fcns,
            firstcaseismedian=False,
            samplemethod=SampleMethod.SOBOL_SALTELLI,
            seed=42,
            singlethreaded=True,
            verbose=False,
            debug=True,
            nstars=512,
            npts=1,
        )

        sim.addInVar(
            name="x1", dist=uniform, distkwargs={"loc": -np.pi, "scale": 2 * np.pi}
        )
        sim.addInVar(
            name="x2", dist=uniform, distkwargs={"loc": -np.pi, "scale": 2 * np.pi}
        )
        sim.addInVar(
            name="x3", dist=uniform, distkwargs={"loc": -np.pi, "scale": 2 * np.pi}
        )

        return sim

    def test_saltelli_sampling_applied(self, ishigami_sim):
        """Test that Saltelli sampling is correctly applied."""
        ishigami_sim.runSim()

        # Check that _saltelli_samples is set
        assert ishigami_sim._saltelli_samples is not None
        assert isinstance(ishigami_sim._saltelli_samples, SaltelliSamples)

        # Check ndraws was adjusted correctly
        expected_ndraws = 512 * (2 + 3 * 1)  # nstars * (2 + ninvars * npts)
        assert ishigami_sim.ndraws == expected_ndraws

    def test_calc_sobol_indices(self, ishigami_sim):
        """Test calcSobolIndices method."""
        ishigami_sim.runSim()
        ishigami_sim.calcSobolIndices("y")

        outvar = ishigami_sim.outvars["y"]

        # Check that indices are set
        assert outvar.sobol_indices is not None
        assert isinstance(outvar.sobol_indices, SobolIndices)
        assert outvar.sensitivity_indices is not None

        # Check first-order indices match analytical values (with tolerance)
        assert outvar.sobol_indices.first_order["x1"] == pytest.approx(
            ISHIGAMI_S1["x1"], abs=0.1
        )
        assert outvar.sobol_indices.first_order["x2"] == pytest.approx(
            ISHIGAMI_S1["x2"], abs=0.1
        )
        # x3 has near-zero first-order effect
        assert outvar.sobol_indices.first_order["x3"] == pytest.approx(
            ISHIGAMI_S1["x3"], abs=0.1
        )

    def test_total_order_indices(self, ishigami_sim):
        """Test total-order indices for Ishigami function."""
        ishigami_sim.runSim()
        ishigami_sim.calcSobolIndices("y")

        outvar = ishigami_sim.outvars["y"]

        # Check total-order indices
        # x1 has interactions with x3, so ST1 > S1
        assert outvar.sobol_indices.total_order["x1"] == pytest.approx(
            ISHIGAMI_ST["x1"], abs=0.15
        )
        # x2 has no interactions, so ST2 = S2
        assert outvar.sobol_indices.total_order["x2"] == pytest.approx(
            ISHIGAMI_ST["x2"], abs=0.1
        )
        # x3 has no main effect but has interaction with x1
        assert outvar.sobol_indices.total_order["x3"] == pytest.approx(
            ISHIGAMI_ST["x3"], abs=0.15
        )

    def test_requires_saltelli_sampling(self):
        """calcSobolIndices should raise error if not using Saltelli sampling."""
        fcns = {
            "preprocess": ishigami_preprocess,
            "run": ishigami_run,
            "postprocess": ishigami_postprocess,
        }

        sim = mc.Sim(
            name="test_error",
            ndraws=100,
            fcns=fcns,
            samplemethod=SampleMethod.SOBOL_RANDOM,  # Not Saltelli
            seed=42,
            verbose=False,
        )
        sim.addInVar("x1", dist=uniform, distkwargs={"loc": 0, "scale": 1})
        sim.addInVar("x2", dist=uniform, distkwargs={"loc": 0, "scale": 1})
        sim.addInVar("x3", dist=uniform, distkwargs={"loc": 0, "scale": 1})
        sim.runSim()

        with pytest.raises(ValueError, match="Saltelli sampling"):
            sim.calcSobolIndices("y")


class TestSobolIndicesNpts:
    """Tests for Sobol' sensitivity with multiple points per arm."""

    def test_npts_greater_than_1(self):
        """Test that npts > 1 works correctly."""
        fcns = {
            "preprocess": ishigami_preprocess,
            "run": ishigami_run,
            "postprocess": ishigami_postprocess,
        }

        sim = mc.Sim(
            name="ishigami_npts_test",
            ndraws=1000,
            fcns=fcns,
            samplemethod=SampleMethod.SOBOL_SALTELLI,
            seed=42,
            verbose=False,
            nstars=128,
            npts=3,  # 3 points per arm
        )

        sim.addInVar(
            "x1", dist=uniform, distkwargs={"loc": -np.pi, "scale": 2 * np.pi}
        )
        sim.addInVar(
            "x2", dist=uniform, distkwargs={"loc": -np.pi, "scale": 2 * np.pi}
        )
        sim.addInVar(
            "x3", dist=uniform, distkwargs={"loc": -np.pi, "scale": 2 * np.pi}
        )

        sim.runSim()

        # Expected: 128 * (2 + 3 * 3) = 128 * 11 = 1408
        assert sim.ndraws == 128 * (2 + 3 * 3)

        sim.calcSobolIndices("y")

        # Indices should still be reasonable
        outvar = sim.outvars["y"]
        assert -0.2 <= outvar.sobol_indices.first_order["x1"] <= 1.0
        assert -0.2 <= outvar.sobol_indices.first_order["x2"] <= 1.0


class TestBootstrapSobolIndices:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_produces_confidence_intervals(self):
        """Test that bootstrap produces confidence intervals."""
        nstars = 256
        samples = saltelli_sampling(nstars, ninvars=2, npts=1, seed=42)

        def model(x):
            return 2 * x[:, 0] + x[:, 1]

        f_A = model(samples.A)
        f_B = model(samples.B)
        f_AB = np.array([model(samples.AB[i, 0]) for i in range(2)])

        indices = bootstrap_sobol_indices(
            f_A=f_A,
            f_B=f_B,
            f_AB=f_AB,
            invar_names=["x1", "x2"],
            confidence_level=0.95,
            n_resamples=100,
            seed=42,
        )

        # Check confidence intervals are set
        assert indices.first_order_conf is not None
        assert indices.total_order_conf is not None

        # Check confidence intervals are reasonable
        for name in ["x1", "x2"]:
            low, high = indices.first_order_conf[name]
            assert low < indices.first_order[name] < high
