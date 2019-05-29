"""
Functions to handle custom containers for the mono reconstruction of LST1
"""

import astropy.units as u
from astropy.units import Quantity
import numpy as np
from ctapipe.core import Container, Field
from ctapipe.image import timing_parameters as time
from ctapipe.image import leakage
from ctapipe.image.cleaning import number_of_islands
from ..reco import utils
from numpy import nan

__all__ = [
    'DL1ParametersContainer',
    'DispContainer',
]

class DL1ParametersContainer(Container):
    """
    TODO: maybe fields could be inherited from ctapipe containers definition
        For now I have not found an elegant way to do so
    """
    intensity = Field(0, 'total intensity (size)')

    x = Field(0 *u.m, 'centroid x coordinate', unit=u.m)
    y = Field(0*u.m, 'centroid x coordinate', unit=u.m)
    r = Field(0*u.m, 'radial coordinate of centroid', unit=u.m)
    phi = Field(0*u.rad, 'polar coordinate of centroid', unit=u.rad)
    length = Field(0*u.m, 'RMS spread along the major-axis', unit=u.m)
    width = Field(0*u.m, 'RMS spread along the minor-axis', unit=u.m)
    psi = Field(0*u.rad, 'rotation angle of ellipse', unit=u.rad)
    skewness = Field(0, 'measure of the asymmetry')
    kurtosis = Field(0, 'measure of the tailedness')
    disp_norm = Field(0*u.m, 'disp_norm [m]', unit=u.m)
    disp_dx = Field(0*u.m, 'disp_dx [m]', unit=u.m)
    disp_dy = Field(0*u.m, 'disp_dy [m]', unit=u.m)
    disp_angle = Field(0*u.rad, 'disp_angle [rad]', unit=u.rad)
    disp_sign = Field(0, 'disp_sign')
    disp_miss = Field(0*u.m, 'disp_miss [m]', unit=u.m)
    src_x = Field(0*u.m, 'source x coordinate in camera frame', unit=u.m)
    src_y = Field(0*u.m, 'source y coordinate in camera frame', unit=u.m)
    time_gradient = Field(0, 'Time gradient in the camera')
    intercept = Field(0, 'Intercept')
    leakage = Field(0, 'Leakage')
    n_islands = Field(0, 'Number of Islands')

    obs_id = Field(0, 'Observation ID')
    event_id = Field(0, 'Event ID')
    gps_time = Field(0, 'GPS time event trigger')

    mc_energy = Field(0*u.TeV, 'Simulated Energy', unit=u.TeV)
    mc_alt = Field(0*u.rad, 'Simulated altitude', unit=u.rad)
    mc_az = Field(0*u.rad, 'Simulated azimuth', unit=u.rad)
    mc_core_x = Field(0*u.m, 'Simulated impact point x position', unit=u.m)
    mc_core_y = Field(0*u.m, 'Simulated impact point y position', unit=u.m)
    mc_h_first_int = Field(0*u.m, 'Simulated first interaction height', unit=u.m)
    mc_type = Field(0, 'Simulated particle type')
    mc_az_tel = Field(0*u.rad, 'Telescope altitude pointing', unit=u.rad)
    mc_alt_tel = Field(0*u.rad, 'Telescope azimuth pointing', unit=u.rad)
    mc_x_max = Field(0*u.g/(u.cm**2), "MC Xmax value", unit=u.g / (u.cm ** 2))
    mc_core_distance = Field(0*u.m, "Distance from the impact point to the telescope", unit=u.m)
    mc_type = Field(0, "MC shower primary ID 0 (gamma), 1(e-),"
                                    "2(mu-), 100*A+Z for nucleons and nuclei,"
                                    "negative for antimatter.")

    hadroness = Field(0, "Hadroness")
    wl = Field(0, "width/length")

    tel_id = Field(0, "Telescope Id")
    tel_pos_x = Field(0, "Telescope x position in the ground")
    tel_pos_y = Field(0, "Telescope y position in the ground")
    tel_pos_z = Field(0, "Telescope z position in the ground")

    def fill_hillas(self, hillas):
        """
        fill Hillas parameters

        hillas: HillasParametersContainer
        # TODO : parameters should not be simply copied but inherited
        (e.g. conserving unit definition)
        """
        for key in hillas.keys():
            self[key] = hillas[key]

    def fill_mc(self, event):
        """
        fill from mc
        """
        try:
            self.mc_energy = event.mc.energy
            self.mc_alt = event.mc.alt
            self.mc_az = event.mc.az
            self.mc_core_x = event.mc.core_x
            self.mc_core_y = event.mc.core_y
            self.mc_h_first_int = event.mc.h_first_int
            # mcType = event.mc. # TODO: find type in event
            self.mc_x_max = event.mc.x_max
            self.mc_alt_tel = event.mcheader.run_array_direction[1]
            self.mc_az_tel = event.mcheader.run_array_direction[0]
            self.mc_type = event.mc.shower_primary_id
        except IndexError:
            print("mc information not filled")

    def fill_event_info(self, event):
        self.gps_time = event.trig.gps_time
        self.obs_id = event.r0.obs_id
        self.event_id = event.r0.event_id

    def get_features(self, features_names):
        return np.array([
            self[k].value
            if isinstance(self[k], Quantity)
            else self[k]
            for k in features_names
        ])

    def set_mc_core_distance(self, event, telescope_id):
        tel_pos = event.inst.subarray.positions[telescope_id]
        distance = np.sqrt(
            (event.mc.core_x - tel_pos[0]) ** 2 +
            (event.mc.core_y - tel_pos[1]) ** 2
        )
        self.mc_core_distance = distance

    def set_disp(self, source_pos, hillas):
        disp = utils.disp_parameters(hillas, source_pos[0], source_pos[1])
        self.disp_norm = disp.norm
        self.disp_dx = disp.dx
        self.disp_dy = disp.dy
        self.disp_angle = disp.angle
        self.disp_sign = disp.sign
        self.disp_miss = disp.miss

    def set_timing_features(self, geom, image, pulse_time, hillas):
        peak_time = Quantity(pulse_time) * u.Unit("ns")
        timepars = time.timing_parameters(geom, image, peak_time, hillas)
        self.time_gradient = timepars.slope.value
        self.intercept = timepars.intercept
    def set_leakage(self, geom, image, clean):
        leakage_c = leakage(geom, image, clean)
        self.leakage = leakage_c.leakage2_intensity

    def set_n_islands(self, geom, clean): 
        n_islands, islands_mask = number_of_islands(geom, clean)
        self.n_islands = n_islands

    def set_telescope_info(self, event, telescope_id):
        self.tel_id = telescope_id
        tel_pos = event.inst.subarray.positions[telescope_id]
        self.tel_pos_x = tel_pos[0] 
        self.tel_pos_y = tel_pos[1] 
        self.tel_pos_z = tel_pos[2] 

    def set_source_camera_position(self, event, telescope_id):
        # sourcepos = utils.cal_cam_source_pos(mc_alt, mc_az,
        #                                      mc_alt_tel, mc_az_tel,
        #                                      focal_length)
        # self.src_x = sourcepos[0]
        # self.src_y = sourcepos[1]
        tel = event.inst.subarray.tel[telescope_id]
        source_pos = utils.get_event_pos_in_camera(event, tel)
        self.src_x = source_pos[0]
        self.src_y = source_pos[1]

    def set_mc_type(self, event):
        self.mc_type = event.mc.shower_primary_id


class DispContainer(Container):
    """
    Disp vector container
    """
    dx = Field(nan, 'x coordinate of the disp_norm vector')
    dy = Field(nan, 'y coordinate of the disp_norm vector')

    angle = Field(nan, 'Angle between the X axis and the disp_norm vector')
    norm = Field(nan, 'Norm of the disp_norm vector')
    sign = Field(nan, 'Sign of the disp_norm')
    miss = Field(nan, 'miss parameter norm')
