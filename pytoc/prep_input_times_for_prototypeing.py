import numpy as np

def gen_poisson_times_parallel(
    num_trials,
    N_pop,
    dt,
    FR,
    std,
    simlen=35000,
    refrac_ms=1.0,
    seed=None,
):
    """
    Generate Poisson spike trains for multiple trials with an absolute refractory period.

    Parameters
    ----------
    num_trials : int
        Number of trials (K).
    N_pop : int
        Number of neurons (N).
    dt : float
        Time step in ms.
    FR : float or ndarray
        Mean firing rate in Hz. Can be scalar, (N,), (T, N), or (K, T, N); will broadcast.
    std : float or ndarray
        Std dev of firing rate noise. Same broadcast rules as FR.
    simlen : int
        Number of time steps (T).
    refrac_ms : float
        Absolute refractory period in ms.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    spikes : np.ndarray, uint8
        Binary spikes with shape (K, T, N) = (num_trials, simlen, N_pop)
    """
    rng = np.random.default_rng(seed)

    K = int(num_trials)
    N = int(N_pop)
    T = int(simlen)
    dt_sec = dt / 1000.0

    # Broadcast FR and std to (K, T, N)
    def _broadcast_to(x, shape):
        if np.isscalar(x):
            return np.full(shape, float(x), dtype=float)
        x = np.asarray(x, dtype=float)
        return np.broadcast_to(x, shape).copy()

    target_shape = (K, T, N)
    FR_KTN  = _broadcast_to(FR,  target_shape)
    std_KTN = _broadcast_to(std, target_shape)

    # Noisy rates (clip at 0 Hz to avoid negative probabilities)
    rates = FR_KTN + std_KTN * rng.standard_normal(target_shape)
    np.clip(rates, 0.0, None, out=rates)

    # Bernoulli probability per bin
    p = rates * dt_sec
    np.clip(p, 0.0, 1.0, out=p)

    # Draw randoms up front
    u = rng.random(target_shape)

    # Allocate spikes and enforce absolute refractory
    spikes = np.zeros(target_shape, dtype=np.uint8)
    refr_samp = int(np.ceil(refrac_ms / dt))
    last = np.full((K, N), -10**9, dtype=np.int32)  # last spike time per (trial, neuron)

    for t in range(T):
        allowed = (t - last) >= refr_samp            # (K, N)
        spk = (u[:, t, :] < p[:, t, :]) & allowed    # (K, N)
        spikes[:, t, :] = spk.astype(np.uint8)
        # update last spike indices
        last = np.where(spk, t, last)

    return spikes