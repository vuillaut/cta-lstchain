from ctapipe.io import event_source
import numpy as np
from .dl0_to_dl1 import get_dl1
from lstchain.reco.utils import camera_to_sky, disp_to_pos
import os
from lstchain.io.lstcontainers import DL1ParametersContainer
import astropy.units as u
import torch
from ...utils.gammalearn import load_model, load_camera_parameters
from ctapipe.io import HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.image.extractor import NeighborPeakWindowSum


allowed_tels = {1, 2, 3, 4}  # select LST1 only
max_events = None  # limit the number of events to analyse in files - None if no limit

threshold = 4094

# Add option to use custom calibration

custom = False

cal = CameraCalibrator(image_extractor=NeighborPeakWindowSum())
serialize_meta = False





def r0_to_dl2(input_filename, features, e_reg, disp_reg, gh_cls, multi_gammalearn_model, output_filename=None):
    """

    Parameters
    ----------
    input_filename
    features
    e_reg
    disp_reg
    gh_cls
    multi_gammalearn_model: end-to-end multitask model
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
    if output_filename is None:
        output_filename = (
                'dl1_' + os.path.basename(input_filename).split('.')[0] + '.h5'
        )

    source = event_source(input_filename)
    source.allowed_tels = allowed_tels
    source.max_events = max_events

    dl1_container = DL1ParametersContainer()
    dl1_container.prefix = ''


    event = next(iter(source))

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



    with HDF5TableWriter(filename=output_filename, group_name='events',
                         overwrite=True, mode='a',
                         add_prefix=True,
                         ) as writer:
        for i, event in enumerate(source):
            cal(event)
            for ii, telescope_id in enumerate(event.r0.tels_with_data):

                dl1_filled = get_dl1(event, telescope_id, dl1_container=dl1_container)
                if dl1_filled is not None:

                    # Some custom def
                    dl1_container.wl = dl1_container.width / dl1_container.length
                    # Log10(Energy) in GeV
                    dl1_container.mc_energy = np.log10(event.mc.energy.value * 1e3)
                    dl1_container.intensity = np.log10(dl1_container.intensity)
                    dl1_container.gps_time = event.trig.gps_time.value

                    foclen = (
                        event.inst.subarray.tel[telescope_id]
                            .optics.equivalent_focal_length
                    )
                    width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
                    length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
                    dl1_container.width = width.value
                    dl1_container.length = length.value

                    X = np.array([u.Quantity(dl1_container.as_dict()[f]).to_value() for f in features])

                    e_pred = e_reg.predict(X.reshape(1, -1))
                    disp_pred = disp_reg.predict(X.reshape(1, -1))[0]
                    X = np.append(X, e_pred)
                    X = np.append(X, disp_pred)

                    # energy
                    event.dl2.energy['lstchain'].energy = 10 ** (e_pred - 3) * u.TeV
                    event.dl2.energy['lstchain'].tel_ids = {telescope_id}
                    event.dl2.energy['lstchain'].prefix = 'lstchain'

                    # direction
                    event.dl2.shower['lstchain'].prefix = 'lstchain'
                    palt = event.mc.tel[telescope_id].altitude_raw * u.rad
                    paz = event.mc.tel[telescope_id].azimuth_raw * u.rad
                    focal_length = event.inst.subarray.tel[telescope_id].optics.equivalent_focal_length
                    src_x, src_y = disp_to_pos(disp_pred[0], disp_pred[1],
                                               dl1_container.x.to_value(),
                                               dl1_container.y.to_value())

                    reco_altaz = camera_to_sky(src_x * u.m, src_y * u.m, focal_length, palt, paz)
                    event.dl2.shower['lstchain'].alt = reco_altaz.alt
                    event.dl2.shower['lstchain'].az = reco_altaz.az

                    # classification
                    event.dl2.classification['lstchain'].prefix = 'lstchain'
                    event.dl2.classification['lstchain'].prediction = gh_cls.predict_proba(X.reshape(1, -1))[0][0]
                    event.dl2.classification['lstchain'].tel_ids = {telescope_id}

                    if width >= 0:
                        is_lstchain_valid = True
                    else:
                        is_lstchain_valid = False

                    event.dl2.energy['lstchain'].is_valid = is_lstchain_valid
                    event.dl2.classification['lstchain'].is_valid = is_lstchain_valid
                    event.dl2.shower['lstchain'].is_valid = is_lstchain_valid

                    ## GammaLearn
                    image = event.dl1.tel[telescope_id].image[0]
                    peakpos = event.dl1.tel[telescope_id].pulse_time[0]

                    data = torch.tensor([image, peakpos], dtype=torch.float).unsqueeze(0)
                    prediction = multi_gammalearn_model(data).squeeze(0).detach().numpy()
                    # particle_prediction = multi_gammalearn_model(data)
                    # particle = torch.max(particle_prediction, 1)[1]

                    event.dl2.energy['gl'].prefix = 'gl'
                    event.dl2.shower['gl'].prefix = 'gl'
                    event.dl2.classification['gl'].prefix = 'gl'
                    event.dl2.energy['gl'].energy = 10 ** prediction[0] * u.TeV
                    event.dl2.shower['gl'].core_x = prediction[1] * u.km
                    event.dl2.shower['gl'].core_y = prediction[2] * u.km
                    event.dl2.shower['gl'].alt = prediction[3] * u.rad
                    event.dl2.shower['gl'].az = prediction[4] * u.rad
                    event.dl2.classification['gl'].prediction = prediction[5]

                    camera = event.inst.subarray.tel[telescope_id].camera
                    writer.write(camera.cam_id, [dl1_container,
                                                 event.dl2.energy['lstchain'],
                                                 event.dl2.shower['lstchain'],
                                                 event.dl2.classification['lstchain'],
                                                 event.dl2.energy['gl'],
                                                 event.dl2.shower['gl'],
                                                 event.dl2.classification['gl'],
                                                 ])


