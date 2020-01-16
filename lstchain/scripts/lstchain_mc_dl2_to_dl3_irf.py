#!/usr/bin/env python

import os
import astropy.units as u
import argparse
import pandas as pd
import numpy as np
import copy

from gammapy.spectrum import cosmic_ray_flux, CrabSpectrum

from protopipe.pipeline.utils import load_config
from protopipe.perf import (CutsOptimisation, CutsDiagnostic, CutsApplicator,
                            IrfMaker, SensitivityMaker)

from lstchain.io.io import dl2_params_lstcam_key
from lstchain.io.io import read_simu_info_merged_hdf5

from astropy.coordinates.angle_utilities import angular_separation


def configuration():
    config = {}
    config['analysis'] = {}
    config['general'] = {}
    config['particle_information'] = {}
    config['column_definition'] = {}

    config['general']['output_table_name'] = 'table_best_cutoff'

    config['column_definition']['mc_energy'] = 'mc_energy'
    config['column_definition']['reco_energy'] = 'reco_energy'
    config['column_definition']['classification_output'] = {}
    config['column_definition']['classification_output']['name'] = 'gammaness'
    config['column_definition']['classification_output']['range'] = [0, 1]
    config['column_definition']['angular_distance_to_the_src'] = 'ang_sep_src'

    config['analysis']['obs_time'] = {
        'value': 0,
        'unit': 'h'
    }

    config['analysis']['thsq_opt'] = {}
    config['analysis']['thsq_opt']['type'] = 'r68'

    config['analysis']['alpha'] = 0.2
    config['analysis']['min_sigma'] = 5
    config['analysis']['min_excess'] = 10
    config['analysis']['bkg_syst'] = 0.05

    config['particle_information']['proton'] = {}
    config['particle_information']['electron'] = {}
    config['particle_information']['proton']['offset_cut'] = 1  # in degrees
    config['particle_information']['electron']['offset_cut'] = 1  # in degrees

    config['analysis']['ereco_binning'] = {}
    config['analysis']['ereco_binning']['emin'] = 0.01
    config['analysis']['ereco_binning']['emax'] = 5
    config['analysis']['ereco_binning']['nbin'] = 21
    config['analysis']['etrue_binning'] = {}
    config['analysis']['etrue_binning']['emin'] = 0.01
    config['analysis']['etrue_binning']['emax'] = 5
    config['analysis']['etrue_binning']['nbin'] = 41

    return config


def get_simu_info(filepath, particle_name, config={}):
    """
    read simu info from file and return config
    """

    if 'particle_information' not in config:
        config['particle_information'] = {}
    if particle_name not in config['particle_information']:
        config['particle_information'][particle_name] = {}
    cfg = config['particle_information'][particle_name]

    simu = read_simu_info_merged_hdf5(filepath)
    cfg['n_events_per_file'] = simu.num_showers * simu.shower_reuse
    cfg['n_files'] = 1
    cfg['e_min'] = simu.energy_range_min
    cfg['e_max'] = simu.energy_range_max
    cfg['gen_radius'] = simu.max_scatter_range
    cfg['diff_cone'] = simu.max_viewcone_radius
    cfg['gen_gamma'] = -simu.spectral_index

    print(particle_name)
    print(cfg)

    return config


def read_and_update_dl2(filepath, tel_id=1, filters=['intensity > 300', 'leakage < 0.2']):
    """
    read DL2 data from lstchain file and update it to be compliant with irf Maker
    """
    data = pd.read_hdf(filepath, key=dl2_params_lstcam_key)
    data = copy.deepcopy(data.query(f'tel_id == {tel_id}'))
    for filter in filters:
        data = copy.deepcopy(data.query(filter))

    # angles are in degrees in protopipe
    data['ang_sep_src'] = pd.Series(angular_separation(data.reco_az.values * u.rad,
                                                       data.reco_alt.values * u.rad,
                                                       data.mc_az.values * u.rad,
                                                       data.mc_alt.values * u.rad,
                                                       ).to(u.deg).value,
                                    index=data.index)

    data['offset'] = pd.Series(angular_separation(data.reco_az.values * u.rad,
                                                  data.reco_alt.values * u.rad,
                                                  data.mc_az_tel.values * u.rad,
                                                  data.mc_alt_tel.values * u.rad,
                                                  ).to(u.deg).value,
                               index=data.index)

    for key in ['mc_alt', 'mc_az', 'reco_alt', 'reco_az', 'mc_alt_tel', 'mc_az_tel']:
        data[key] = np.rad2deg(data[key])

    return data


def main():

    parser = argparse.ArgumentParser(description='Make performance files')

    parser.add_argument(
        '--obs_time',
        dest='obs_time',
        type=float,
        default=50,
        help='Observation time in hours'
    )

    parser.add_argument('--dl2_gamma', '-g',
                        dest='dl2_gamma_filename',
                        type=str,
                        required=True,
                        help='path to the gamma dl2 file'
                        )

    parser.add_argument('--dl2_proton', '-p',
                        dest='dl2_proton_filename',
                        type=str,
                        required=True,
                        help='path to the proton dl2 file'
                        )

    parser.add_argument('--dl2_electron', '-e',
                        dest='dl2_electron_filename',
                        type=str,
                        required=True,
                        help='path to the electron dl2 file'
                        )

    parser.add_argument('--outdir', '-o',
                        dest='outdir',
                        type=str,
                        default='.',
                        help="Output directory"
                        )

    args = parser.parse_args()
    paths = {}
    paths['gamma'] = args.dl2_gamma_filename
    paths['proton'] = args.dl2_proton_filename
    paths['electron'] = args.dl2_electron_filename

    # Read configuration file
    # cfg = load_config(args.config_file)
    cfg = configuration()

    cfg['analysis']['obs_time']['value'] = args.obs_time

    cfg['general']['outdir'] = args.outdir

    # Create output directory if necessary
    outdir = os.path.join(cfg['general']['outdir'], 'irf_ThSq_{}_Time{:.2f}{}'.format(
        cfg['analysis']['thsq_opt']['type'],
        cfg['analysis']['obs_time']['value'],
        cfg['analysis']['obs_time']['unit'])
                          )
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load data
    particles = ['gamma', 'electron', 'proton']
    evt_dict = dict()  # Contain DL2 file for each type of particle
    for particle in particles:
        infile = paths[particle]
        evt_dict[particle] = read_and_update_dl2(infile)
        cfg = get_simu_info(infile, particle, config=cfg)

    # Apply offset cut to proton and electron
    for particle in ['electron', 'proton']:
        evt_dict[particle] = evt_dict[particle].query('offset <= {}'.format(
            cfg['particle_information'][particle]['offset_cut'])
        )

    # Add required data in configuration file for future computation
    for particle in particles:
        # cfg['particle_information'][particle]['n_files'] = \
        #     len(np.unique(evt_dict[particle]['obs_id']))
        cfg['particle_information'][particle]['n_simulated'] = \
            cfg['particle_information'][particle]['n_files'] * cfg['particle_information'][particle][
                'n_events_per_file']

    # Define model for the particles
    model_dict = {'gamma': CrabSpectrum('hegra').model,
                  'proton': cosmic_ray_flux,
                  'electron': cosmic_ray_flux}

    # Reco energy binning
    cfg_binning = cfg['analysis']['ereco_binning']
    ereco = np.logspace(np.log10(cfg_binning['emin']),
                        np.log10(cfg_binning['emax']),
                        cfg_binning['nbin'] + 1) * u.TeV

    # Handle theta square cut optimisation
    # (compute 68 % containment radius PSF if necessary)
    thsq_opt_type = cfg['analysis']['thsq_opt']['type']
    # if thsq_opt_type in 'fixed':
    #     thsq_values = np.array([cfg['analysis']['thsq_opt']['value']]) * u.deg
    #     print('Using fixed theta cut: {}'.format(thsq_values))
    # elif thsq_opt_type in 'opti':
    #     thsq_values = np.arange(0.05, 0.40, 0.01) * u.deg
    #     print('Optimising theta cut for: {}'.format(thsq_values))
    if thsq_opt_type is not 'r68':
        raise ValueError("only r68 supported at the moment")
    elif thsq_opt_type in 'r68':
        print('Using R68% theta cut')
        print('Computing...')
        cfg_binning = cfg['analysis']['ereco_binning']
        ereco = np.logspace(np.log10(cfg_binning['emin']),
                            np.log10(cfg_binning['emax']),
                            cfg_binning['nbin'] + 1) * u.TeV
        radius = 68

        thsq_values = list()
        for ibin in range(len(ereco) - 1):
            emin = ereco[ibin]
            emax = ereco[ibin + 1]

            energy_query = 'reco_energy > {} and reco_energy <= {}'.format(
                emin.value, emax.value
            )
            data = evt_dict['gamma'].query(energy_query).copy()

            min_stat = 0
            if len(data) <= min_stat:
                print('  ==> Not enough statistics:')
                print('To be handled...')
                thsq_values.append(0.3)
                continue
                # import sys
                # sys.exit()

            psf = np.percentile(data['offset'], radius)
            psf_err = psf / np.sqrt(len(data))

            thsq_values.append(psf)
        thsq_values = np.array(thsq_values) * u.deg
        # Set 0.05 as a lower value
        idx = np.where(thsq_values.value < 0.05)
        thsq_values[idx] = 0.05 * u.deg
        print('Using theta cut: {}'.format(thsq_values))

    # Cuts optimisation
    print('### Finding best cuts...')
    cut_optimiser = CutsOptimisation(
        config=cfg,
        evt_dict=evt_dict,
        verbose_level=0
    )

    # Weight events
    print('- Weighting events...')
    cut_optimiser.weight_events(
        model_dict=model_dict,
        colname_mc_energy=cfg['column_definition']['mc_energy']
    )

    # Find best cutoff to reach best sensitivity
    print('- Estimating cutoffs...')
    cut_optimiser.find_best_cutoff(energy_values=ereco, angular_values=thsq_values)

    # Save results and auxiliary data for diagnostic
    print('- Saving results to disk...')
    cut_optimiser.write_results(
        outdir, '{}.fits'.format(cfg['general']['output_table_name']),
        format='fits'
    )

    # Cuts diagnostic
    print('### Building cut diagnostics...')
    cut_diagnostic = CutsDiagnostic(config=cfg, indir=outdir)
    cut_diagnostic.plot_optimisation_summary()
    cut_diagnostic.plot_diagnostics()

    # Apply cuts and save data
    print('### Applying cuts to data...')
    cut_applicator = CutsApplicator(config=cfg, evt_dict=evt_dict, outdir=outdir)
    cut_applicator.apply_cuts()

    # Irf Maker
    print('### Building IRF...')
    irf_maker = IrfMaker(config=cfg, evt_dict=evt_dict, outdir=outdir)
    irf_maker.build_irf()

    # Sensitivity maker
    print('### Estimating sensitivity...')
    sensitivity_maker = SensitivityMaker(config=cfg, outdir=outdir)
    sensitivity_maker.load_irf()
    sensitivity_maker.estimate_sensitivity()


if __name__ == '__main__':
    main()
