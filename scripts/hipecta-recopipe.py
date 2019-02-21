import numpy as np
from ctapipe.utils import get_dataset_path
from ctapipe.io import event_source
from ctapipe.io import HDF5TableWriter
import pandas as pd
from lstchain.reco.dl1_to_dl2 import apply_models
from lstchain.io.containers import DL1ParametersContainer

from lstchain.reco import dl0_to_dl1
from lstchain.reco import dl1_to_dl2
from sklearn.externals import joblib
import argparse
import os
from distutils.util import strtobool

### PARAMETERS - TODO: use a yaml config file


parser = argparse.ArgumentParser(description="Reconstruct events")

# Required arguments
parser.add_argument('--datafile', '-f', type=str,
                    dest='datafile',
                    help='path to the file with simtelarray events',
                    default=get_dataset_path('gamma_test_large.simtel.gz'))

parser.add_argument('--pathmodels', '-p', action='store', type=str,
                     dest='path_models',
                     help='Path where to find the trained RF',
                     default='./trained_models')

# Optional argument
parser.add_argument('--dl1_dir', '-dl1', action='store', type=lambda x: bool(strtobool(x)),
                    dest='dl1_dir',
                    help='Optional. If given, reconstructed dl1 parameters will be stored in the given path',
                    default=None)

parser.add_argument('--outdir', '-o', action='store', type=str,
                     dest='outdir',
                     help='Path where to store the reco dl2 events',
                     default='./dl2_results')

parser.add_argument('--maxevents', '-x', action='store', type=int,
                     dest='max_events',
                     help='Maximum number of events to analyze',
                     default=None)

args = parser.parse_args()



# def r0_to_dl2(models_path, models_features, dl0_filename=get_dataset_path('gamma_test_large.simtel.gz'), outdir = '.'):
#     """
#     Chain r0 to dl2
#     Save the extracted dl2 parameters in output_filename
#
#     Parameters
#     ----------
#     dl0_filename: str - path to input file, default: `gamma_test_large.simtel.gz`
#     output_filename: str - path to output file, default: `./` + basename(input_filename)
#
#     Returns
#     -------
#
#     """
#     import os
#     output_filename = outdir + '/dl2_' + os.path.basename(dl0_filename).split('.')[0] + '.h5'
#
#     source = event_source(dl0_filename)
#     source.allowed_tels = allowed_tels
#     source.max_events = max_events
#
#     reg_energy = joblib.load(models_path + "/reg_energy.sav" )
#     reg_disp = joblib.load(models_path + "/reg_disp.sav")
#     cls_gh = joblib.load(models_path + "/cls_gh.sav")
#
#     dl1_container = DL1ParametersContainer()
#     features = DL1ParametersContainer().keys()
#     data = pd.DataFrame(columns=features)
#
#
#     # data.to_hdf('test.h5', key='events')
#     data = pd.read_hdf('test.h5')
#     dl2 = apply_models(data[models_features].dropna(), models_features, cls_gh, reg_energy, reg_disp)
#
#     dl2.to_hdf(output_filename)


if __name__ == '__main__':

    features = ['intensity',
                # 'time_gradient',
                'width',
                'length',
                'wl',
                'phi',
                'psi',
                # 'mc_core_distance'
                ]

    dl1_outdir = 'dl1_data/' if args.dl1_dir is not None else './'

    dl1_filename = dl1_outdir + '/dl1_' + os.path.basename(args.datafile).split('.')[0] + '.h5'
    os.makedirs(dl1_outdir, exist_ok=True)
    dl0_to_dl1.max_events = args.max_events
    dl0_to_dl1.hipecta_r0_to_dl1(args.datafile, dl1_outdir)
    dl1 = pd.read_hdf(dl1_filename).dropna()

    reg_energy = joblib.load(args.path_models + "/reg_energy.sav" )
    reg_disp = joblib.load(args.path_models + "/reg_disp.sav")
    cls_gh = joblib.load(args.path_models + "/cls_gh.sav")

    dl2 = dl1_to_dl2.apply_models(dl1, features, cls_gh, reg_energy, reg_disp)
    os.makedirs(args.outdir, exist_ok=True)
    outfile = args.outdir + '/dl2_' + os.path.basename(args.datafile).split('.')[0] + '.h5'
    dl2.to_hdf(outfile, key='events/')


    if args.dl1_dir is None:
        os.remove(dl1_filename)

