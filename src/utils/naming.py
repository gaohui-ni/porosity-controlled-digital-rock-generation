"""Canonical naming helpers shared by generation and evaluation workflows."""


def phi_tag(phi: float) -> str:
    """Return a stable porosity tag without losing significant decimals."""
    return (
        f"{float(phi):.4f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )
