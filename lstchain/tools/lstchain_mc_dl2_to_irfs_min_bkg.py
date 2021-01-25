"""
Example for using pyirf to calculate IRFS and sensitivity from EventDisplay DL2 fits
files produced from the root output by this script:

https://github.com/Eventdisplay/Converters/blob/master/DL2/generate_DL2_file.py
"""
import logging
import operator

import numpy as np
from astropy import table
import astropy.units as u
from astropy.io import fits


from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
)
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution

from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.cut_optimization import optimize_gh_cut

from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
    background_2d,
)

from pyirf.io import (
    create_aeff2d_hdu,
    create_psf_table_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
    create_background_2d_hdu,
)

from lstchain.io.io import read_dl2_to_pyirf

log = logging.getLogger("lstchain MC DL2 to IRF")

import argparse
from pathlib import Path


from lstchain.reco.utils import filter_events

parser = argparse.ArgumentParser(description="MC DL2 to IRF")

# Required arguments
parser.add_argument('--gamma-dl2', '-g',
                    type=Path,
                    dest='gamma_file',
                    help='Path to the dl2 gamma file',
                    )

parser.add_argument('--proton-dl2', '-p',
                    type=Path,
                    dest='proton_file',
                    help='Path to the dl2 proton file',
                    )

parser.add_argument('--electron-dl2', '-e',
                    type=Path,
                    dest='electron_file',
                    help='Path to the dl2 electron file',
                    )

parser.add_argument('--outfile', '-o', action='store', type=Path,
                    dest='outfile',
                    help='Path where to save IRF FITS file',
                    default=f'irf.fits.gz'
                    )

# Optional arguments
parser.add_argument('--config', '-c', action='store', type=Path,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )


args = parser.parse_args()

T_OBS = 50 * u.hour

# scaling between on and off region.
# Make off region 10 times larger than on region for better
# background statistics

ALPHA = 0.1

# Radius to use for calculating bg rate
MAX_BG_RADIUS = 1 * u.deg
MAX_GH_CUT_EFFICIENCY = 0.9
GH_CUT_EFFICIENCY_STEP = 0.01

# gh cut used for first calculation of the binned theta cuts
INITIAL_GH_CUT_EFFICENCY = 0.4

MIN_THETA_CUT = 0.1 * u.deg
MAX_THETA_CUT = 0.5 * u.deg

MIN_ENERGY = 20 * u.GeV
MAX_ENERGY = 20.05 * u.TeV

# min number of background event to consider the sensitivity calculation acceptable
MIN_PROTON_EVENT_PER_BIN = 0
MIN_BKG_EVENT_PER_BIN = 0

N_BIN_PER_DECADE = 5

# source position
source_alt = 70 * u.deg
source_az = 180 * u.deg


particles = {
    "gamma": {
        "file": args.gamma_file,
        "target_spectrum": CRAB_HEGRA,
    },
    "proton": {
        "file": args.proton_file,
        "target_spectrum": IRFDOC_PROTON_SPECTRUM,
    },
    "electron": {
        "file": args.electron_file,
        "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
    },
}

background_list = ['proton', 'electron']
# background_list = ['proton']



filters = {'tel_id': [1, 1],
           'intensity': [50, np.inf],
           'leakage_intensity_width_2': [0, 0.2],
          }




from tqdm import tqdm
from pyirf.cuts import evaluate_binned_cut, calculate_percentile_cut
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.binning import create_histogram_table, bin_center
from astropy.table import QTable

def max_gammaness_cut(background, reco_energy_bins, min_n_bkg, gammaness_bins=10000):
    """
    find the max gammaness cut for keep a n_bkg > min_n_bkg
    """
    gh_cuts = []
    for ii in range(len(reco_energy_bins)-1):
        energy_mask = (background['reco_energy']>=reco_energy_bins[ii]) & (background['reco_energy']<reco_energy_bins[ii+1])
        theta_mask = (background['theta'] < MAX_BG_RADIUS)
        hist, bin_edges = np.histogram(background[energy_mask & theta_mask]['gh_score'], bins=gammaness_bins)
        cuts_with_enough_bkg = (np.cumsum(hist[::-1])[::-1] >= min_n_bkg).nonzero()[0]
        if len(cuts_with_enough_bkg) > 0:
            highest_gammaness_cut = bin_edges[cuts_with_enough_bkg[-1]]
        else:
            highest_gammaness_cut = 0
        gh_cuts.append(highest_gammaness_cut)
        
    return gh_cuts



def optimize_gh_cut(
    signal,
    background,
    protons,
    reco_energy_bins,
    gh_cut_efficiencies,
    theta_cuts,
    op=operator.ge,
    background_radius=1 * u.deg,
    alpha=1,
    progress=True,
    **kwargs
):
    """
    Optimize the gh-score cut in every energy bin of reconstructed energy
    for best sensitivity.
    This procedure is EventDisplay-like, since it only applies a
    pre-computed theta cut and then optimizes only the gamma/hadron separation
    cut.
    Parameters
    ----------
    signal: astropy.table.QTable
        event list of simulated signal events.
        Required columns are `theta`, `reco_energy`, 'weight', `gh_score`
        No directional (theta) or gamma/hadron cut should already be applied.
    background: astropy.table.QTable
        event list of simulated background events.
        Required columns are `reco_source_fov_offset`, `reco_energy`,
        'weight', `gh_score`.
        No directional (theta) or gamma/hadron cut should already be applied.
    reco_energy_bins: astropy.units.Quantity[energy]
        Bins in reconstructed energy to use for sensitivity computation
    gh_cut_efficiencies: np.ndarray[float, ndim=1]
        The cut efficiencies to scan for best sensitivity.
    theta_cuts: astropy.table.QTable
        cut definition of the energy dependent theta cut,
        e.g. as created by ``calculate_percentile_cut``
    op: comparison function with signature f(a, b) -> bool
        The comparison function to use for the gamma hadron score.
        Returning true means an event passes the cut, so is not discarded.
        E.g. for gammaness-like score, use `operator.ge` (>=) and for a
        hadroness-like score use `operator.le` (<=).
    background_radius: astropy.units.Quantity[angle]
        Radius around the field of view center used for background rate
        estimation.
    alpha: float
        Size ratio of off region / on region. Will be used to
        scale the background rate.
    progress: bool
        If True, show a progress bar during cut optimization
    **kwargs are passed to ``calculate_sensitivity``
    """

    # we apply each cut for all reco_energy_bins globally, calculate the
    # sensitivity and then lookup the best sensitivity for each
    # bin independently

    signal_selected_theta = evaluate_binned_cut(
        signal['theta'], signal['reco_energy'], theta_cuts,
        op=operator.le,
    )

    sensitivities = []
    gh_cuts = []
    for efficiency in tqdm(gh_cut_efficiencies, disable=not progress):

        # calculate necessary percentile needed for
        # ``calculate_percentile_cut`` with the correct efficiency.
        # Depends on the operator, since we need to invert the
        # efficiency if we compare using >=, since percentile is
        # defines as <=.
        if op(-1, 1): # if operator behaves like "<=", "<" etc:
            percentile = 100 * efficiency
            fill_value = signal['gh_score'].min()
        else: # operator behaves like ">=", ">"
            percentile = 100 * (1 - efficiency)
            fill_value = signal['gh_score'].max()

        gh_cut = calculate_percentile_cut(
            signal['gh_score'], signal['reco_energy'],
            bins=reco_energy_bins,
            fill_value=fill_value, percentile=percentile,
        )
        
        # make sure we have more than MIN_BKG events per reco_energy bin to compute IRFs
        max_gh_cut_p = max_gammaness_cut(protons, reco_energy_bins, MIN_PROTON_EVENT_PER_BIN)
        max_gh_cut_bkg = max_gammaness_cut(background, reco_energy_bins, MIN_BKG_EVENT_PER_BIN)
        max_gh_cut = np.min([max_gh_cut_p, max_gh_cut_bkg], axis=0)
        gh_cut['cut'] = np.min([gh_cut['cut'], max_gh_cut], axis=0)
        gh_cuts.append(gh_cut)

        # apply the current cut
        signal_selected = evaluate_binned_cut(
            signal["gh_score"], signal["reco_energy"], gh_cut, op,
        ) & signal_selected_theta

        background_selected = evaluate_binned_cut(
            background["gh_score"], background["reco_energy"], gh_cut, op,
        )

        # create the histograms
        signal_hist = create_histogram_table(
            signal[signal_selected], reco_energy_bins, "reco_energy"
        )

        background_hist = estimate_background(
            events=background[background_selected],
            reco_energy_bins=reco_energy_bins,
            theta_cuts=theta_cuts,
            alpha=alpha,
            background_radius=background_radius
        )

        sensitivity = calculate_sensitivity(
            signal_hist, background_hist, alpha=alpha,
            **kwargs,
        )
        
        if (background_hist['n'] < 10).all():
            import pdb; pdb.set_trace();
        
        
        sensitivities.append(sensitivity)

    best_cut_table = QTable()
    best_cut_table["low"] = reco_energy_bins[0:-1]
    best_cut_table["center"] = bin_center(reco_energy_bins)
    best_cut_table["high"] = reco_energy_bins[1:]
    best_cut_table["cut"] = np.nan

    best_sensitivity = sensitivities[0].copy()
    for bin_id in range(len(reco_energy_bins) - 1):
        sensitivities_bin = np.array([s["relative_sensitivity"][bin_id] for s in sensitivities])
        
        if not np.all(np.isnan(sensitivities_bin)):
            # nanargmin won't return the index of nan entries
            best = np.nanargmin(sensitivities_bin)
        else:
            # if all are invalid, just use the first one
            best = 0

        best_sensitivity[bin_id] = sensitivities[best][bin_id]
        best_cut_table["cut"][bin_id] = gh_cuts[best]["cut"][bin_id]

    return best_sensitivity, best_cut_table





def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pyirf").setLevel(logging.DEBUG)

    for particle_type, p in particles.items():
        log.info(f"Simulated {particle_type.title()} Events:")
        p["events"], p["simulation_info"] = read_dl2_to_pyirf(p["file"])
        
        p["events"] = filter_events(p["events"], filters)
        
        print('=====', particle_type, '=====')
        # p["events"]["particle_type"] = particle_type

        p["simulated_spectrum"] = PowerLaw.from_simulation(p["simulation_info"], T_OBS)
        p["events"]["weight"] = calculate_event_weights(
            p["events"]["true_energy"], p["target_spectrum"], p["simulated_spectrum"]
        )
        for prefix in ('true', 'reco'):
            k = f"{prefix}_source_fov_offset"
            p["events"][k] = calculate_source_fov_offset(p["events"], prefix=prefix)

        # calculate theta / distance between reco and assuemd source positoin
        # we handle only ON observations here, so the assumed source pos
        # is the pointing position
        p["events"]["theta"] = calculate_theta(
            p["events"],
            assumed_source_az=source_az,
            assumed_source_alt=source_alt,
        )
        log.info(p["simulation_info"])
        log.info("")

    gammas = particles["gamma"]["events"]
    # background table composed of both electrons and protons
    background = table.vstack(
        [particles[p]["events"] for p in background_list]
    )
    protons = particles['proton']['events']

    INITIAL_GH_CUT = np.quantile(gammas['gh_score'], (1 - INITIAL_GH_CUT_EFFICENCY))
    log.info(f"Using fixed G/H cut of {INITIAL_GH_CUT} to calculate theta cuts")

    # event display uses much finer bins for the theta cut than
    # for the sensitivity
    theta_bins = add_overflow_bins(
        create_bins_per_decade(MIN_ENERGY, MAX_ENERGY, N_BIN_PER_DECADE,)
    )

    # theta cut is 68 percent containmente of the gammas
    # for now with a fixed global, unoptimized score cut
    mask_theta_cuts = gammas["gh_score"] >= INITIAL_GH_CUT
    theta_cuts = calculate_percentile_cut(
        gammas["theta"][mask_theta_cuts],
        gammas["reco_energy"][mask_theta_cuts],
        bins=theta_bins,
        min_value=MIN_THETA_CUT,
        fill_value=MAX_THETA_CUT,
        max_value=MAX_THETA_CUT,
        percentile=68,
    )

    # same number of bins per decade than EventDisplay
    sensitivity_bins = add_overflow_bins(create_bins_per_decade(MIN_ENERGY, MAX_ENERGY, bins_per_decade=N_BIN_PER_DECADE))

    log.info("Optimizing G/H separation cut for best sensitivity")
    gh_cut_efficiencies = np.arange(
        GH_CUT_EFFICIENCY_STEP,
        MAX_GH_CUT_EFFICIENCY + GH_CUT_EFFICIENCY_STEP / 2,
        GH_CUT_EFFICIENCY_STEP
    )
    sensitivity_step_2, gh_cuts = optimize_gh_cut(
        gammas,
        background,
        protons,
        reco_energy_bins=sensitivity_bins,
        gh_cut_efficiencies=gh_cut_efficiencies,
        op=operator.ge,
        theta_cuts=theta_cuts,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )

    # now that we have the optimized gh cuts, we recalculate the theta
    # cut as 68 percent containment on the events surviving these cuts.
    log.info('Recalculating theta cut for optimized GH Cuts')
    for tab in (gammas, background):
        tab["selected_gh"] = evaluate_binned_cut(
            tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
        )

    theta_cuts_opt = calculate_percentile_cut(
        gammas[gammas['selected_gh']]["theta"],
        gammas[gammas['selected_gh']]["reco_energy"],
        theta_bins,
        percentile=68,
        fill_value=MAX_THETA_CUT,
        max_value=MAX_THETA_CUT,
        min_value=MIN_THETA_CUT,
    )

    gammas["selected_theta"] = evaluate_binned_cut(
        gammas["theta"], gammas["reco_energy"], theta_cuts_opt, operator.le
    )
    gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]

    # calculate sensitivity
    signal_hist = create_histogram_table(
        gammas[gammas["selected"]], bins=sensitivity_bins
    )
    background_hist = estimate_background(
        background[background["selected_gh"]],
        reco_energy_bins=sensitivity_bins,
        theta_cuts=theta_cuts_opt,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )
    sensitivity = calculate_sensitivity(
        signal_hist, background_hist, alpha=ALPHA
    )

    # scale relative sensitivity by Crab flux to get the flux sensitivity
    spectrum = particles['gamma']['target_spectrum']
    for s in (sensitivity_step_2, sensitivity):
        s["flux_sensitivity"] = (
            s["relative_sensitivity"] * spectrum(s['reco_energy_center'])
        )

    log.info('Calculating IRFs')
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
        fits.BinTableHDU(sensitivity_step_2, name="SENSITIVITY_STEP_2"),
        fits.BinTableHDU(theta_cuts, name="THETA_CUTS"),
        fits.BinTableHDU(theta_cuts_opt, name="THETA_CUTS_OPT"),
        fits.BinTableHDU(gh_cuts, name="GH_CUTS"),
    ]

    masks = {
        "": gammas["selected"],
        "_NO_CUTS": slice(None),
        "_ONLY_GH": gammas["selected_gh"],
        "_ONLY_THETA": gammas["selected_theta"],
    }

    # binnings for the irfs
    true_energy_bins = add_overflow_bins(
        create_bins_per_decade(MIN_ENERGY, MAX_ENERGY, N_BIN_PER_DECADE)
    )
    reco_energy_bins = add_overflow_bins(
        create_bins_per_decade(MIN_ENERGY, MAX_ENERGY, N_BIN_PER_DECADE)
    )
    fov_offset_bins = [0, 0.6] * u.deg
    source_offset_bins = np.arange(0, 1 + 1e-4, 1e-3) * u.deg
    energy_migration_bins = np.geomspace(0.2, 5, 200)

    for label, mask in masks.items():
        effective_area = effective_area_per_energy(
            gammas[mask],
            particles["gamma"]["simulation_info"],
            true_energy_bins=true_energy_bins,
        )
        hdus.append(
            create_aeff2d_hdu(
                effective_area[..., np.newaxis],  # add one dimension for FOV offset
                true_energy_bins,
                fov_offset_bins,
                extname="EFFECTIVE_AREA" + label,
            )
        )
        edisp = energy_dispersion(
            gammas[mask],
            true_energy_bins=true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            migration_bins=energy_migration_bins,
        )
        hdus.append(
            create_energy_dispersion_hdu(
                edisp,
                true_energy_bins=true_energy_bins,
                migration_bins=energy_migration_bins,
                fov_offset_bins=fov_offset_bins,
                extname="ENERGY_DISPERSION" + label,
            )
        )

    bias_resolution = energy_bias_resolution(
        gammas[gammas["selected"]], true_energy_bins,
    )
    ang_res = angular_resolution(gammas[gammas["selected_gh"]], true_energy_bins,)
    psf = psf_table(
        gammas[gammas["selected_gh"]],
        true_energy_bins,
        fov_offset_bins=fov_offset_bins,
        source_offset_bins=source_offset_bins,
    )

    background_rate = background_2d(
        background[background['selected_gh']],
        reco_energy_bins,
        fov_offset_bins=np.arange(0, 11) * u.deg,
        t_obs=T_OBS,
    )

    hdus.append(create_background_2d_hdu(
        background_rate,
        reco_energy_bins,
        fov_offset_bins=np.arange(0, 11) * u.deg,
    ))
    hdus.append(create_psf_table_hdu(
        psf, true_energy_bins, source_offset_bins, fov_offset_bins,
    ))
    hdus.append(create_rad_max_hdu(
        theta_cuts_opt["cut"][:, np.newaxis], theta_bins, fov_offset_bins
    ))
    hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))
    hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))

    log.info('Writing outputfile')
    fits.HDUList(hdus).writeto(args.outfile, overwrite=True)


if __name__ == "__main__":
    main()
