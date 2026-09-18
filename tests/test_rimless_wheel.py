import numpy as np
import pytest

from integrators import rk4
from models import rimless_wheel as model


@pytest.fixture
def params():
    return {"gravity": 9.81, "length": 1.0, "mass": 0.2, "N": 8, "Incline": np.pi / 24}


def test_impact_requires_forward_crossing(params):
    contact = np.pi / params["N"] + params["Incline"]
    assert model.detect_event([contact - 0.01, 1], [contact, 1], params)
    assert not model.detect_event([contact - 0.01, 1], [contact, -1], params)
    assert not model.detect_event([contact, 1], [contact + 0.01, 1], params)


def test_reset_preserves_fractional_values_and_input(params):
    state = np.array([1, 2])
    reset = model.reset_impact(state, params)
    np.testing.assert_allclose(reset, [params["Incline"] - np.pi / 8, np.sqrt(2)])
    np.testing.assert_array_equal(state, [1, 2])


def test_roa_classifies_every_column(params):
    theta, omega = np.meshgrid([-0.2, 0.0, 0.2], [0.0, 1.0])
    result = model.compute_roa(
        theta, omega, 2, np.array([0.0, 0.01]), 0.01, 1e-5, params, rk4, model
    )
    np.testing.assert_array_equal(result, [[0, 4, 4], [4, 4, 4]])


def test_return_map_matches_energy_balance(params):
    alpha = np.pi / params["N"]
    incline = params["Incline"]
    velocity = 1.5
    expected = np.cos(2 * alpha) * np.sqrt(
        velocity**2
        + 4 * params["gravity"] / params["length"] * np.sin(alpha) * np.sin(incline)
    )
    actual = model.next_impact_velocity(
        velocity, params, 0.001, 1e-6, rk4, alpha - incline
    )
    assert actual == pytest.approx(expected, abs=1e-5)


def test_sweep_returns_fractions_without_mutating_parameters(params, monkeypatch):
    import assignment_1_sweep as sweep

    original = params.copy()
    seen = []

    def compute_roa(*args):
        seen.append(args[6].copy())
        return np.array([[0, 3], [3, 4]])

    monkeypatch.setattr(sweep.model, "compute_roa", compute_roa)
    monkeypatch.setattr(
        sweep.model, "next_impact_velocity", lambda velocity, *args: velocity
    )
    monkeypatch.setattr(sweep.model, "estimate_floquet", lambda *args: 0.5)
    monkeypatch.setattr(sweep, "save_roa_map", lambda *args: None)
    values, fractions, floquet, maps = sweep.sweep_parameter(
        params, "Incline", [0.1, 0.2], 2
    )
    np.testing.assert_allclose(values, [0.1, 0.2])
    np.testing.assert_allclose(fractions, [0.5, 0.5])
    np.testing.assert_allclose(floquet, [0.5, 0.5])
    assert len(maps) == 2
    assert [item["Incline"] for item in seen] == [0.1, 0.2]
    assert params == original
