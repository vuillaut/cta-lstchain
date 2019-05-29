from ctapipe.io import event_source
import numpy as np
from lstchain.reco.dl0_to_dl1 import get_dl1
import os
from lstchain.io.lstcontainers import DL1ParametersContainer
import astropy.units as u
import torch
from ctapipe.io import HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.image.extractor import NeighborPeakWindowSum
from lstchain.reco.utils import camera_to_sky, disp_to_pos
from sklearn.externals import joblib
from astropy.coordinates import SkyCoord
from ctapipe.coordinates import NominalFrame, CameraFrame
from lstchain.reco.utils import horizon_frame
import pandas as pd

import sys
sys.path.insert(0,'../utils')
from gammalearn import load_model, load_camera_parameters


allowed_tels = {1, 2, 3, 4}  # select LST1 only
max_events = None  # limit the number of events to analyse in files - None if no limit

threshold = 4094

# Add option to use custom calibration

custom = False

cal = CameraCalibrator(image_extractor=NeighborPeakWindowSum())
serialize_meta = False

softmax = torch.nn.Softmax()

exp_path = '/gpfs/MUST-DATA/glearn/Data/experiments/'
exp_name = 'R_0173_end'
checkpoint = '60'
# camera_model_path = '/Users/thomasvuillaume/Work/GammaLearn/GammaLearn/share/camera_parameters.h5'
camera_model_path = '/uds_data/glearn/Software/GammaLearn/share/camera_parameters.h5'
model = load_model(exp_path, exp_name, checkpoint, camera_model_path)
model_type = 'regression'


def r0_to_dl2(input_filename, model, model_type, e_reg, disp_reg, cls_gh, features, output_filename=None):
    """

    Parameters
    ----------
    input_filename
    features
    e_reg
    disp_reg
    gh_cls
    model: pytorch model
    model_type:
        'regression':
            predict:
            - energy (in log(E/TeV))
            - xCore (in km)
            - yCore (in km)
            - altitude (in rad)
            - azimuth (in rad)
        'classification':
            predict:
            - particle type
        'full':
            predict:
            - energy (in log(E/TeV))
            - xCore (in km)
            - yCore (in km)
            - altitude (in rad)
            - azimuth (in rad)
            - particle type

    output_filename

    Returns
    -------

    """

    assert model_type in ['regression', 'classification', 'full']

    if output_filename is None:
        output_filename = (
                'dl2_' + os.path.basename(input_filename).split('.')[0] + '.h5'
        )

    source = event_source(input_filename)
    source.allowed_tels = allowed_tels
    source.max_events = max_events

    dl1_container = DL1ParametersContainer()
    dl1_container.prefix = ''



    with HDF5TableWriter(filename=output_filename, group_name='events',
                         overwrite=True, mode='a',
                         add_prefix=True,
                         ) as writer:
        for i, event in enumerate(source):
            if i % 100 == 0:
                print(i)
            #             cal(event)
            for ii, telescope_id in enumerate(event.r0.tels_with_data):

                dl1_filled = get_dl1(event, telescope_id, dl1_container=dl1_container)

                if dl1_filled is not None:
                    # Some custom def
                    dl1_container.wl = dl1_container.width / dl1_container.length
                    # Log10(Energy) in GeV
                    dl1_container.mc_energy = np.log10(event.mc.energy.value * 1e3)
                    dl1_container.intensity = np.log10(dl1_container.intensity)
                    dl1_container.gps_time = event.trig.gps_time.value

                    foclen = (event.inst.subarray.tel[telescope_id].optics.equivalent_focal_length)
                    width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
                    length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
                    dl1_container.width = width.value
                    dl1_container.length = length.value

                    ## GammaLearn
                    image = event.dl1.tel[telescope_id].image[0]
                    peakpos = event.dl1.tel[telescope_id].pulse_time[0]

                    data = torch.tensor([image, peakpos], dtype=torch.float).unsqueeze(0)

                    event.dl2.classification['gl'].prefix = 'gl'
                    event.dl2.energy['gl'].prefix = 'gl'
                    event.dl2.shower['gl'].prefix = 'gl'

                    if model_type in ['regression', 'full']:
                        prediction = model(data).squeeze(0).detach().numpy()
                        event.dl2.energy['gl'].energy = 10 ** prediction[0] * u.TeV
                        event.dl2.shower['gl'].core_x = prediction[1] * u.km
                        event.dl2.shower['gl'].core_y = prediction[2] * u.km
                        event.dl2.shower['gl'].alt = prediction[3] * u.rad
                        event.dl2.shower['gl'].az = prediction[4] * u.rad
                        event.dl2.energy['gl'].is_valid = True
                        event.dl2.shower['gl'].is_valid = True

                        if model_type == 'full':
                            soft = softmax(torch.tensor(prediction[5:]))
                            event.dl2.classification['gl'].prediction = soft[0].item()
                            event.dl2.classification['gl'].is_valid = True

                    elif model_type == 'classification':

                        particle_prediction = model(data)
                        particle = torch.max(particle_prediction, 1)[1]


                    camera = event.inst.subarray.tel[telescope_id].camera
                    writer.write(camera.cam_id, [dl1_container,
                                                 #                                                  event.dl2.energy['lstchain'],
                                                 #                                                  event.dl2.shower['lstchain'],
                                                 #                                                  event.dl2.classification['lstchain'],
                                                 event.dl2.energy['gl'],
                                                 event.dl2.shower['gl'],
                                                 event.dl2.classification['gl'],
                                                 ])

    sub = event.inst.subarray
    sub.to_table().write(
        output_filename,
        path="/instrument/subarray/layout",
        serialize_meta=serialize_meta,
        overwrite=True
    )

    sub.to_table(kind='optics').write(
        output_filename,
        path='/instrument/telescope/optics',
        append=True,
        serialize_meta=serialize_meta
    )
    for telescope_type in sub.telescope_types:
        ids = set(sub.get_tel_ids_for_type(telescope_type)).intersection(allowed_tels)
        if len(ids) > 0:  # only write if there is a telescope with this camera
            tel_id = list(ids)[0]
            camera = sub.tel[tel_id].camera
            camera.to_table().write(
                output_filename,
                path=f'/instrument/telescope/camera/{camera}',
                append=True,
                serialize_meta=serialize_meta,
            )

    with HDF5TableWriter(
            filename=output_filename,
            group_name='header',
            mode='a',
            overwrite=True,
    ) as writer:
        writer.write('mc', event.mcheader)

    telescope_id = list(event.r0.tels_with_data)[0]
    palt = event.mcheader.run_array_direction[1]
    paz = event.mcheader.run_array_direction[0]
    focal_length = event.inst.subarray.tel[telescope_id].optics.equivalent_focal_length




    df = pd.read_hdf(output_filename, key='events/LSTCam')

    X = df[features]
    mask_valid = np.isfinite(X).sum(axis=1) == X.shape[1]
    # if np.isfinite(X).all():
    X_valid = X[mask_valid]
    e_pred = e_reg.predict(X_valid)
    disp_pred = disp_reg.predict(X_valid)

    src_x, src_y = disp_to_pos(disp_pred[:, 0], disp_pred[:, 1], X_valid.x.values, X_valid.y.values)
    reco_altaz = camera_to_sky(src_x * u.m, src_y * u.m, focal_length, palt, paz)

    X_valid = np.c_[X_valid.to_numpy(), e_pred]
    X_valid = np.c_[X_valid, disp_pred]
    prediction = cls_gh.predict_proba(X_valid)
    gammaness = prediction[:, 0]
    hadroness = prediction[:, 1]

    lstchain_energy = np.nan * np.ones(len(X))
    lstchain_alt = np.nan * np.ones(len(X))
    lstchain_az = np.nan * np.ones(len(X))
    lstchain_gammaness = np.nan * np.ones(len(X))
    lstchain_valid = np.zeros(len(X))

    lstchain_energy[mask_valid] = e_pred
    lstchain_alt[mask_valid] = reco_altaz.alt.rad
    lstchain_az[mask_valid] = reco_altaz.az.rad
    lstchain_gammaness[mask_valid] = gammaness
    lstchain_valid[mask_valid] = 1

    df.assign(lstchain_energy=pd.Series(lstchain_energy))
    df.assign(lstchain_alt=pd.Series(lstchain_alt))
    df.assign(lstchain_az=pd.Series(lstchain_az))
    df.assign(lstchain_gammaness=pd.Series(lstchain_gammaness))
    df.assign(lstchain_valid=pd.Series(lstchain_valid))

    df.to_hdf()