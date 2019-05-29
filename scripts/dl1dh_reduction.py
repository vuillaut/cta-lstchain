import argparse
from lstchain.reco import dl0_to_dl1
import os

parser = argparse.ArgumentParser(description="R0 to DL1")

parser.add_argument('--infile', '-f', type=str,
                    dest='infile',
                    help='path to the DL1DH file ',
                    )

parser.add_argument('--outdir', '-o', action='store', type=str,
                    dest='outdir',
                    help='Path where to store the reco dl2 events',
                    default='./dl1_data/')

args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(args.outdir, exist_ok=True)

    dl0_to_dl1.allowed_tels = {1, 2, 3, 4}
    dl0_to_dl1.max_events = 10
    output_filename = args.outdir + '/dl1_' + os.path.basename(args.infile).rsplit('.', 1)[0] + '.h5'

    dl0_to_dl1.dl1dh_to_dl1(args.infile, output_filename=output_filename)

