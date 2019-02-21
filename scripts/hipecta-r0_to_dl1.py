
from ctapipe.utils import get_dataset_path
import os
import argparse
from lstchain.reco.dl0_to_dl1 import hipecta_r0_to_dl1

### PARAMETERS - TODO: use a yaml config file


parser = argparse.ArgumentParser(description="R0 to DL1")

parser.add_argument('--infile', '-f', type=str,
                    dest='infile',
                    help='path to the file with simtelarray events',
                    default=get_dataset_path('gamma_test_large.simtel.gz'))

parser.add_argument('--outdir', '-o', action='store', type=str,
                    dest='outdir',
                    help='Path where to store the reco dl2 events',
                    default='./dl1_data/')

args = parser.parse_args()



if __name__ == '__main__':
    os.makedirs(args.outdir, exist_ok=True)

    hipecta_r0_to_dl1(input_filename=args.infile, outdir=args.outdir)