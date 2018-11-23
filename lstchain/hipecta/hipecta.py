"""
set of functions to make the reconstruction using hipecta
"""

import numpy as np
from ctapipe.utils import get_dataset_path
from ctapipe.io import event_source
from ctapipe.io import HDF5TableWriter

from ..io.containers import DL1ParametersContainer
from ..reco import utils

from hipecta.image import get_hillas_parameters_container
from hipecta.image import TelescopeReco
import os


### PARAMETERS - TODO: use a yaml config file


allowed_tels = {1}  # select LST1 only
max_events = None  # limit the number of events to analyse in files - None if no limit

reco = TelescopeReco(wavelet_threshold=1, hillas_threshold_signal_tel=50)

channel = 0


def get_dl1(event, telescope_id):
    """
    Return a DL1ParametersContainer of extracted features from a calibrated event

    Parameters
    ----------
    event: ctapipe event container
    telescope_id:

    Returns
    -------
    DL1ParametersContainer
    """
    dl1_container = DL1ParametersContainer()

    tabHillas, isGoodEvent = reco.process(telescope_id, event, True)

    if isGoodEvent:
        hillas = get_hillas_parameters_container(tabHillas)
        dl1_container.fill_hillas(hillas)
        dl1_container.set_mc_core_distance(event, telescope_id)
        dl1_container.set_source_camera_position(event, telescope_id)
        dl1_container.set_disp([dl1_container.src_x, dl1_container.src_y], hillas)
        dl1_container.wl = dl1_container.width / dl1_container.length
        dl1_container.mc_energy = np.log10(event.mc.energy.value * 1e3)  # Log10(Energy) in GeV
        dl1_container.intensity = np.log10(dl1_container.intensity)
        dl1_container.gps_time = event.trig.gps_time.value

        foclen = event.inst.subarray.tel[telescope_id].optics.equivalent_focal_length
        w = np.rad2deg(np.arctan2(dl1_container.width, foclen))
        l = np.rad2deg(np.arctan2(dl1_container.length, foclen))
        dl1_container.width = w.value
        dl1_container.length = l.value

        return dl1_container
    else:
        return None




def r0_to_dl1(input_filename=get_dataset_path('gamma_test_large.simtel.gz'), output_filename=None, allowed_tels={1}):
    """
    Chain r0 to dl1
    Save the extracted dl1 parameters in output_filename

    Parameters
    ----------
    reco: `hipecta.image.TelescopeReco()`
    input_filename: str - path to input file, default: `gamma_test_large.simtel.gz`
    output_filename: str - path to output file, default: `./` + basename(input_filename)
    """
    output_filename = 'dl1_' + os.path.basename(input_filename).split('.')[0] + '.h5' if output_filename is None \
        else output_filename

    source = event_source(input_filename, allowed_tels=allowed_tels)


    with HDF5TableWriter(filename=output_filename, group_name='events', overwrite=True) as writer:

        for i, event in enumerate(source):
            if i% 100 == 0: print(i)

            for ii, telescope_id in enumerate(event.r0.tels_with_data):
                camera = event.inst.subarray.tel[telescope_id].camera  # Camera geometry

                dl1_container = get_dl1(event, telescope_id)
                if dl1_container is not None:
                    particle_name = utils.guess_type(input_filename)
                    dl1_container.mc_type = utils.particle_number(particle_name)
                    dl1_container.hadroness = 1 if dl1_container.mc_type == 1 else 0

                    writer.write(camera.cam_id, [dl1_container])
