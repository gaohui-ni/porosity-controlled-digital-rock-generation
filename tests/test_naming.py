from src.utils.naming import phi_dir, phi_tag


def test_phi_directory_uses_evaluator_compatible_name():
    assert phi_tag(0.11) == "0p11"
    assert phi_dir(0.11) == "phi0p11"
    assert phi_dir(0.1289) == "phi0p1289"
