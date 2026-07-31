def quality_gate_message(label, total, successful, min_success_rate, allow_partial):
    """Return a warning message or raise when an evaluation has too few results."""
    if not 0.0 <= min_success_rate <= 1.0:
        raise ValueError("min_success_rate must be between 0 and 1.")
    if total <= 0:
        raise RuntimeError(f"{label}: no input samples were found.")
    if successful <= 0:
        raise RuntimeError(f"{label}: all {total} samples failed.")

    rate = successful / total
    if rate >= min_success_rate:
        return None

    message = (
        f"{label}: success rate {rate:.1%} ({successful}/{total}) is below "
        f"the required {min_success_rate:.1%}."
    )
    if allow_partial:
        return message
    raise RuntimeError(message)
