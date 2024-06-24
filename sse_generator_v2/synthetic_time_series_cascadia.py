import numpy as np

from sse_generator_v2.artificial_noise_cascadia import artificial_noise_full_randomized_v3
from sse_generator_v2.synthetic_sse_cascadia import synthetic_sses_v2


def synthetic_time_series_real_gaps_extended_v5(n_samples, n_selected_stations, window_length=60, **kwargs):
    """Equivalent to the formulation used by SSEgenerator, but adapted to denonising. What's new:
    - we consider the whole sequence 2007-2023 to generate the artificial noise
    - we consider more realistic synthetic slow slip events (see the doc of the new function synthetic_sses_v2)
    - we also return the synthetic slow slip (sigmoids).
    - we did not keep the 'detection' mode, as the processing is a mix of detection and characterization,
        since we also allow for negative cases (displacement=0 along the window)."""
    reference_period = (2007, 2023)
    max_n_disloc = kwargs.pop('max_n_disloc', 3)
    aspect_ratio = kwargs.pop('aspect_ratio', 1 / 2)
    correct_latlon = kwargs.pop('correct_latlon', False)
    noise_windows, station_codes, station_coordinates = artificial_noise_full_randomized_v3(n_samples,
                                                                                            n_selected_stations,
                                                                                            window_length,
                                                                                            reference_period,
                                                                                            p=kwargs['p'])

    transients, random_durations, synthetic_displacement, catalogue = synthetic_sses_v2(n_samples, window_length,
                                                                                        station_codes,
                                                                                        station_coordinates,
                                                                                        max_n_disloc=max_n_disloc,
                                                                                        aspect_ratio=aspect_ratio,
                                                                                        correct_latlon=correct_latlon,
                                                                                        **kwargs)

    synthetic_data = noise_windows + transients
    synthetic_data = np.nan_to_num(synthetic_data, nan=0.)  # NaNs are put back to zero

    rnd_idx = np.random.permutation(len(synthetic_displacement))  # not really necessary, but we do it anyway
    synthetic_data = synthetic_data[rnd_idx]
    random_durations = random_durations[rnd_idx]
    transients = transients[rnd_idx]
    catalogue = [catalogue[i] for i in rnd_idx]  # catalogue[rnd_idx]
    synthetic_displacement = [synthetic_displacement[i] for i in rnd_idx]  # synthetic_displacement[rnd_idx]
    return synthetic_data, random_durations, catalogue, synthetic_displacement, transients, station_codes, station_coordinates
