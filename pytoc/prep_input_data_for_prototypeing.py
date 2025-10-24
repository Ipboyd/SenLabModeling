import numpy as np
import scipy.io
import os

def _spike_generator_batch(rates_TNK, dt=0.1, t_ref=2.0, t_ref_rel=8.0, rec=2.0, seed=None):
    """
    Vectorized Poisson spike generator with absolute and relative refractory periods.

    Parameters
    ----------
    rates_TNK : np.ndarray
        Firing rates in Hz with shape (T, N, K) = (time, neurons, trials).
    dt : float
        Timestep in ms.
    t_ref : float
        Absolute refractory period in ms.
    t_ref_rel : float
        Relative refractory time constant (ms) controlling recovery.
    rec : float
        Sharpness (exponent) of the recovery function.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    spikes_KTN : np.ndarray, dtype=uint8
        Binary spike trains with shape (K, T, N) = (trials, time, neurons).
    """
    rng = np.random.default_rng(seed)
    dt_sec = dt / 1000.0

    T, N, K = rates_TNK.shape
    spikes_TNK = np.zeros((T, N, K), dtype=np.uint8)

    # Convert refractory periods to samples
    t_ref_samp = max(0, int(round(t_ref / dt)))
    t_rel_samp = max(1, int(round(t_ref_rel / dt)))  # at least 1 to avoid 0^rec

    # Length of the post-spike window for applying recovery weights.
    # Past this, recovery is effectively ~1, so we clamp.
    post_win = max(1, t_ref_samp + t_rel_samp)
    tw = np.arange(post_win + 1, dtype=np.int32)

    # Schaette-style recovery function:
    # w(deltat) = 0 for deltat < t_ref; otherwise ((deltat - t_ref)^rec) / (((deltat - t_ref)^rec) + (t_rel^rec))
    with np.errstate(divide='ignore', invalid='ignore'):
        w = ((tw - t_ref_samp) ** rec) / ( ((tw - t_ref_samp) ** rec) + (t_rel_samp ** rec) )
        w[tw < t_ref_samp] = 0.0
        w = np.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)

    # Track last spike time per (neuron, trial)
    last = np.full((N, K), -10**9, dtype=np.int32)  # "minus infinity" in samples

    # Time loop (vectorized over neurons × trials)
    for t in range(T):
        delta = t - last                                   # (N, K)
        delta_clipped = np.clip(delta, 0, post_win)        # (N, K)
        weight = w[delta_clipped]                          # (N, K), fancy indexed

        # Effective rate after refractory
        lam_eff = rates_TNK[t] * weight                    # (N, K)

        # Poisson test with Bernoulli per time bin
        x = rng.random((N, K))
        spk = (x < dt_sec * lam_eff)
        spikes_TNK[t] = spk.astype(np.uint8)

        # Update last spike time
        last = np.where(spk, t, last)

    # Return as (K, T, N): (trials, time, neurons)
    return np.transpose(spikes_TNK, (2, 0, 1))


def gen_poisson_inputs_parallel(
    num_trials,
    loc_num=None,
    label="on",
    t_ref=1,
    t_ref_rel=1,
    rec=2.0,
    scale_factor=1.0,
    layer_name="1-channel-paper",
    dt=0.1,
    n_locations=24,
    seed=None
):
    """
    Generate Poisson spike inputs for ALL trials in one call.

    Parameters
    ----------
    num_trials : int
        Number of trials to generate (will truncate to what's available in the .mat).
    loc_num : int or None
        1-based location index to extract. If None, uses ALL locations (concatenated in time).
    label : str
        Stimulus label used to load the correct .mat file (e.g., 'on' or 'off').
    t_ref, t_ref_rel, rec : floats
        Refractory parameters (ms, ms, and exponent respectively).
    scale_factor : float
        Multiplier applied to the rates before sampling.
    layer_name : str
        Subfolder used to compose the path to the .mat file.
    dt : float
        Timestep in ms (used for converting refractory periods to samples).
    n_locations : int
        Number of locations packed along the first dimension of the .mat (default 24).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    spikes_KTN : np.ndarray, dtype=uint8
        Binary spike trains with shape (num_trials, T, N).
    """
    # Build path & load rates
    matfile_path = os.path.join(
        "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Single-Channel/Model/Model-Core/Model-Main/run",
        layer_name,
        "solve",
    )
    filename = os.path.join(matfile_path, f"IC_spks_{label}.mat")
    data = scipy.io.loadmat(filename)

    # Expect data['spks'] shape: (time*locations, neurons, trials)
    spks = np.asarray(data["spks"])  # (TL, N, K_all)
    TL, N, K_all = spks.shape

    # Trim to requested number of trials
    K = min(num_trials, K_all)
    spks = spks[:, :, :K]  # (TL, N, K)



    rates_TNK = spks  # (TL, N, K)


    # Scale rates if desired
    rates_TNK = rates_TNK * float(scale_factor)

    # Generate spikes for all trials in parallel
    spikes_KTN = _spike_generator_batch(
        rates_TNK, dt=dt, t_ref=t_ref, t_ref_rel=t_ref_rel, rec=rec, seed=seed
    )

    return spikes_KTN