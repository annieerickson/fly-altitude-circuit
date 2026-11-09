import os 
import numpy as np
from scipy.special import binom

def check_dir(directory):
    """
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    else:
        pass


def to_deg(x):
    return x*(180.0/np.pi)

def LegendrePolynomials(N_pts,N_pol,n_deriv):
    """
    N_const: degree stitch together subsequent wingbeats
    N_deriv: number of derivatives to computer for legendre basis
    """
    L_basis = np.zeros((N_pts,N_pol,n_deriv))
    x_basis = np.linspace(-1.0,1.0,N_pts,endpoint=True)
    for i in range(n_deriv):
        if i==0:
            # Legendre basis:
            for n in range(N_pol):
                if n==0:
                    L_basis[:,n,i] = 1.0
                elif n==1:
                    L_basis[:,n,i] = x_basis
                else:
                    for k in range(n+1):
                        L_basis[:,n,i] += (1.0/np.power(2.0,n))*np.power(binom(n,k),2)*np.multiply(np.power(x_basis-1.0,n-k),np.power(x_basis+1.0,k))
        else:
            # Derivatives:
            for n in range(N_pol):
                if n>=i:
                    L_basis[:,n,i] = n*L_basis[:,n-1,i-1]+np.multiply(x_basis,L_basis[:,n-1,i])
    return L_basis

def reconstruct_trace_coef_matrix(cur_coeffs, deriv_num=0, n_deriv=2, N_pts=100):
    """
    return trace for vector for a coefficients (often avg coeff vector)
    if reading in multiple wbs need to transpose to coef by # wbs
    """
    L_basis = LegendrePolynomials(N_pts, len(cur_coeffs), n_deriv)
    return np.matmul(L_basis[:,:,deriv_num], cur_coeffs)


def mean_wingbeat_traces(wb, movie, angle, side, diff_from_avg=True):
    baseline, other = [], []        # other = diff coefs (diff_from_avg) or stim traces
    for fly in movie.keys():
        if not diff_from_avg:
            pre = wb[f"{fly}/angle/{angle}/{side}/wb/pre"][:]
            stim = wb[f"{fly}/angle/{angle}/{side}/wb/stim"][:]
            baseline.append(np.nanmean(np.vstack(pre), axis=0))
            other.append(np.nanmean(np.vstack(stim), axis=0))
        else:
            base_fly, diff_fly = [], []
            for m in movie[fly].keys():
                pre = movie[f"{fly}/{m}/angle/{angle}/{side}/a/pre"][:]
                stim = movie[f"{fly}/{m}/angle/{angle}/{side}/a/stim"][:]
                base_fly.append(pre)
                diff_fly.append(stim - np.nanmean(pre, axis=0))
            baseline.append(np.nanmean(np.vstack(base_fly), axis=0))
            other.append(np.nanmean(np.vstack(diff_fly), axis=0))

    baseline, other = np.vstack(baseline), np.vstack(other)
    if diff_from_avg:                       # still in Legendre coefficients
        base_coef = np.nanmean(baseline, axis=0)
        base_trace = reconstruct_trace_coef_matrix(base_coef)
        stim_trace = reconstruct_trace_coef_matrix(base_coef + np.nanmean(other, axis=0))
    else:
        base_trace = np.nanmean(baseline, axis=0)
        stim_trace = np.nanmean(other, axis=0)
    return base_trace, stim_trace

KEY_POINT = {"phi": "min", "theta": "max", "eta": "min", "xi": "max"}

def _combine(left, right, mode):
    n = min(len(left), len(right))                       
    l, r = np.asarray(left[:n]), np.asarray(right[:n])
    return l + r if mode == "L+R" else np.abs(l - r)

def key_point_fly_summaries(movie, angle, wing_combine="L+R", baseline_subtract="per_stimulus"):
    """Return pre, stimulation, and movie-difference fly means in radians.
    """
    reduce = np.nanmax if KEY_POINT[angle] == "max" else np.nanmin
    pre_means, stim_means, changes = [], [], []
    for fly in movie.keys():
        pre_fly, stim_fly = {"L": [], "R": []}, {"L": [], "R": []}
        pre_movie_means, stim_movie_means, movie_changes = [], [], []
        for m in movie[fly].keys():
            base = f"{fly}/{m}/angle/{angle}"
            arr = {(s, ph): movie[f"{base}/{s}/wb/{ph}"][:] for s in "LR" for ph in ("pre", "stim", "post")}
            if any(len(a) == 0 for a in arr.values()):    # skip trials with any empty array
                continue
            pre = {s: reduce(arr[(s, "pre")], axis=1) for s in "LR"}
            stim = {s: reduce(arr[(s, "stim")], axis=1) for s in "LR"}
            for s in "LR":
                pre_fly[s].extend(pre[s])
                stim_fly[s].extend(stim[s])
            if baseline_subtract == "per_stimulus":
                pre_mean = np.mean(_combine(pre["L"], pre["R"], wing_combine))
                stim_mean = np.mean(_combine(stim["L"], stim["R"], wing_combine))
                movie_changes.append(stim_mean - pre_mean)
                pre_movie_means.append(pre_mean)
                stim_movie_means.append(stim_mean)
        if baseline_subtract == "per_stimulus":
            pre_means.append(np.mean(pre_movie_means))
            stim_means.append(np.mean(stim_movie_means))
            changes.append(np.mean(movie_changes))
        else:                                             # "per_fly": pool all wingbeats
            pre_means.append(np.mean(_combine(pre_fly["L"], pre_fly["R"], wing_combine)))
            stim_means.append(np.mean(_combine(stim_fly["L"], stim_fly["R"], wing_combine)))
            changes.append(stim_means[-1] - pre_means[-1])
    return np.array(pre_means), np.array(stim_means), np.array(changes)


def key_point_fly_means(movie, angle, wing_combine="L+R", baseline_subtract="per_stimulus"):
    """Compatibility interface returning the two per-fly period means."""
    pre, stim, _ = key_point_fly_summaries(movie, angle, wing_combine, baseline_subtract)
    return pre, stim
