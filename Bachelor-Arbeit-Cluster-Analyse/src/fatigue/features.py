import numpy as np

from fatigue.io import load_mat, extract_dual_axis_parameters


G = 9.81


def compute_duty_factor(contact_times, flight_times):
    """
    Compute mean duty factor.

    DF = stance / (stance + flight)

    Handles unequal array lengths.
    """

    ct = np.asarray(contact_times, dtype=float)
    ft = np.asarray(flight_times, dtype=float)

    # gleiche Länge erzwingen
    n = min(len(ct), len(ft))

    ct = ct[:n]
    ft = ft[:n]

    print("ContactTimes:", len(ct))
    print("FlightTimes:", len(ft))
    print("Used strides:", n)

    df = ct / (ct + ft)

    df = df[np.isfinite(df)]

    if df.size == 0:
        raise ValueError("Duty factor array empty")

    return float(np.mean(df))


def compute_step_frequency(contact_times, flight_times):
    """
    Compute step frequency from contact and flight times.

    SF = 1 / (contact + flight)
    """

    ct = np.asarray(contact_times, dtype=float)
    ft = np.asarray(flight_times, dtype=float)

    n = min(len(ct), len(ft))

    ct = ct[:n]
    ft = ft[:n]

    step_time = ct + ft

    sf = 1.0 / step_time

    sf = sf[np.isfinite(sf)]

    return float(np.mean(sf))


def compute_sf_norm_from_times(contact_times, flight_times, leg_length):
    """
    Compute normalized step frequency.

    SF_norm = SF * sqrt(l0/g)
    """

    SF = compute_step_frequency(contact_times, flight_times)

    SF_norm = SF * np.sqrt(leg_length / G)

    return float(SF_norm)


def dual_axis_from_mat(mat_path, leg_length):
    """
    Extract DF and SF_norm from MAT file.
    """

    mat = load_mat(mat_path)

    p = extract_dual_axis_parameters(mat)

    DF = compute_duty_factor(
        p["ContactTimes"],
        p["FlightTimes"]
    )

    SF_norm = compute_sf_norm_from_times(
        p["ContactTimes"],
        p["FlightTimes"],
        leg_length
    )

    return dict(
        DF=DF,
        SF_norm=SF_norm
    )