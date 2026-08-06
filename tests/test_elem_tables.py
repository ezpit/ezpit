import numpy as np
import pytest

from ezpit import elem_tables as et


# --- Table consistency -------------------------------------------------------

def test_aff_table_shapes_match():
    assert len(et.AFF_ELEMENTS) == len(et.AFF_PARAMETERS)
    assert all(len(row) == 11 for row in et.AFF_PARAMETERS)


def test_compton_table_shapes_match():
    n = len(et.COMPTON_ELEMENTS)
    assert len(et.COMPTON_ATOMIC_NUMBERS) == n
    assert len(et.COMPTON_PARAMETERS) == n
    assert all(len(row) == 11 for row in et.COMPTON_PARAMETERS)


def test_compton_atomic_numbers_are_sequential():
    assert et.COMPTON_ATOMIC_NUMBERS == list(range(1, len(et.COMPTON_ELEMENTS) + 1))


def test_element_names_are_unique():
    assert len(set(et.AFF_ELEMENTS)) == len(et.AFF_ELEMENTS)
    assert len(set(et.COMPTON_ELEMENTS)) == len(et.COMPTON_ELEMENTS)


# --- get_aff_scattering_factors ---------------------------------------------

def test_get_aff_scattering_factors_shape():
    factors = et.get_aff_scattering_factors(['Co', 'O'])
    assert factors.shape == (2, 11)


def test_get_aff_scattering_factors_matches_table():
    idx = et.AFF_ELEMENTS.index('Fe')
    expected = np.array(et.AFF_PARAMETERS[idx])
    np.testing.assert_array_equal(et.get_aff_scattering_factors(['Fe'])[0], expected)


def test_get_aff_scattering_factors_is_case_insensitive():
    np.testing.assert_array_equal(
        et.get_aff_scattering_factors(['co']),
        et.get_aff_scattering_factors(['Co']),
    )


def test_get_aff_scattering_factors_supports_ions():
    idx = et.AFF_ELEMENTS.index('Fe2+')
    expected = np.array(et.AFF_PARAMETERS[idx])
    np.testing.assert_array_equal(et.get_aff_scattering_factors(['Fe2+'])[0], expected)


def test_get_aff_scattering_factors_unknown_raises():
    with pytest.raises(KeyError):
        et.get_aff_scattering_factors(['Zz'])


# --- get_compton_scattering_factors -----------------------------------------

def test_get_compton_scattering_factors_returns_factors_and_numbers():
    factors, numbers = et.get_compton_scattering_factors(['Co', 'O'])
    assert factors.shape == (2, 11)
    assert numbers == [27, 8]


def test_get_compton_scattering_factors_matches_table():
    idx = et.COMPTON_ELEMENTS.index('Ni')
    factors, numbers = et.get_compton_scattering_factors(['Ni'])
    np.testing.assert_array_equal(factors[0], np.array(et.COMPTON_PARAMETERS[idx]))
    assert numbers[0] == et.COMPTON_ATOMIC_NUMBERS[idx]


def test_get_compton_scattering_factors_is_case_insensitive():
    lower, _ = et.get_compton_scattering_factors(['fe'])
    upper, _ = et.get_compton_scattering_factors(['Fe'])
    np.testing.assert_array_equal(lower, upper)


def test_get_compton_scattering_factors_unknown_raises():
    with pytest.raises(ValueError, match="No Compton data for atom: Zz"):
        et.get_compton_scattering_factors(['Zz'])


# --- get_compton_parameter_only ---------------------------------------------

def test_get_compton_parameter_only_shape():
    table = et.get_compton_parameter_only()
    assert table.shape == (len(et.COMPTON_ELEMENTS), 11)


def test_get_compton_parameter_only_returns_copy():
    table = et.get_compton_parameter_only()
    table[0, 0] = -12345.0
    assert et.get_compton_parameter_only()[0, 0] != -12345.0
