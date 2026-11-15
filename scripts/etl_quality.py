def check_quality(total_rows, accepted_rows, rejected_rows, threshold):
    """Validate row accounting and determine publication eligibility."""

    if total_rows <= 0:
        raise ValueError("Quality failure: input dataset is empty")

    if total_rows != accepted_rows + rejected_rows:
        raise ValueError("Quality failure: row accounting mismatch")

    if not 0 <= threshold <= 1:
        raise ValueError("Quality failure: invalid rejection threshold")

    rejection_rate = rejected_rows / total_rows

    result = {
        "total_rows": total_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "rejection_rate": rejection_rate,
        "threshold": threshold,
        "publication_allowed": rejection_rate <= threshold
    }

    return result
