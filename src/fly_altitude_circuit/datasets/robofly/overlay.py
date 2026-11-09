"""Aerodynamic overlay scene; VTK imports only on rendering."""

import os
import numpy as np

def render_overlay(self, panel, destination):
    from ...paths import ensure_output_path
    from .lollipop_figure import Lollipop
    destination = ensure_output_path(destination)
    scaling_factor = panel['mean_force_scale']
    plot_mean_forces = panel['plot_mean_forces']
    plot_lollipops = panel['plot_lollipops']
    Rwing_on = panel['right_wing_on']
    exp_name = 'baseline_stim'
    save_loc = str(destination.parent) + os.sep
    extension = '.png'
    dpi = panel['dpi']
    '\n        baseline and stim, L and R wings\n        #note: cv2.imwrite can not write svgs so using other libraries\n        Can save as pdf or png but not editable vectors (screencap of 3d scene)\n        '
    LP = Lollipop()
    LP.n_skip = panel['chord_sample_stride']
    LP.capture_scale = panel['capture_scale']
    LP.camera_focal = panel['camera_focal']
    LP.Renderer()
    LP.ConstructModel(True)
    s_thorax = np.array([np.cos(-35.0 / 180.0 * np.pi / 2.0), 0.0, np.sin(-35.0 / 180.0 * np.pi / 2.0), 0.0, 0.0, 0.0, 0.0])
    s_head = np.array([np.cos(-5.0 / 180.0 * np.pi / 2.0), 0.0, np.sin(-5.0 / 180.0 * np.pi / 2.0), 0.0, 0.55, 0.0, 0.42])
    s_abdomen = np.array([np.cos(-70.0 / 180.0 * np.pi / 2.0), 0.0, np.sin(-70.0 / 180.0 * np.pi / 2.0), 0.0, 0.0, 0.0, -0.1])
    body_scale = panel['body_scale']
    body_clr = [tuple(panel['body_color'])]
    LP.SetBodyColor(body_clr)
    LP.SetBodyScale(body_scale)
    LP.SetBodyState(s_thorax, s_head, s_abdomen)
    n_pts = 100
    wing_length = 2.0
    joint_L = np.array([0.0, 0.5, 0.0])
    joint_R = np.array([0.0, -0.5, 0.0])
    LE_pt = 0.1
    TE_pt = -0.2
    for i in [0, 2]:
        if i == 0:
            m_clr = (0.5, 0.5, 0.5)
            m_clr_2 = (0, 0, 0)
        else:
            m_clr = (1.0, 0.0, 0.0)
            m_clr_2 = (1.0, 0.0, 0.0)
        theta_L = self.wingkin_SRF_list_means[i][0, :]
        eta_L = self.wingkin_SRF_list_means[i][1, :]
        phi_L = self.wingkin_SRF_list_means[i][2, :]
        xi_L = self.wingkin_SRF_list_means[i][3, :]
        theta_R = self.wingkin_SRF_list_means[i + 1][0, :]
        eta_R = self.wingkin_SRF_list_means[i + 1][1, :]
        phi_R = self.wingkin_SRF_list_means[i + 1][2, :]
        xi_R = self.wingkin_SRF_list_means[i + 1][3, :]
        n_pts = xi_R.shape[0]
        FX_mean_L = self.FT_SRF_list_means[i][0, :] * self.F_scaling_list_means[i] * scaling_factor
        FY_mean_L = self.FT_SRF_list_means[i][1, :] * self.F_scaling_list_means[i] * scaling_factor
        FZ_mean_L = self.FT_SRF_list_means[i][2, :] * self.F_scaling_list_means[i] * scaling_factor
        MX_mean_L = self.FT_SRF_list_means[i][3, :] * self.M_scaling_list_means[i] * scaling_factor
        MY_mean_L = self.FT_SRF_list_means[i][4, :] * self.M_scaling_list_means[i] * scaling_factor
        MZ_mean_L = self.FT_SRF_list_means[i][5, :] * self.M_scaling_list_means[i] * scaling_factor
        FX_mean_R = self.FT_SRF_list_means[i + 1][0, :] * self.F_scaling_list_means[i + 1] * scaling_factor
        FY_mean_R = self.FT_SRF_list_means[i + 1][1, :] * self.F_scaling_list_means[i + 1] * scaling_factor
        FZ_mean_R = self.FT_SRF_list_means[i + 1][2, :] * self.F_scaling_list_means[i + 1] * scaling_factor
        MX_mean_R = self.FT_SRF_list_means[i + 1][3, :] * self.M_scaling_list_means[i + 1] * scaling_factor
        MY_mean_R = self.FT_SRF_list_means[i + 1][4, :] * self.M_scaling_list_means[i + 1] * scaling_factor
        MZ_mean_R = self.FT_SRF_list_means[i + 1][5, :] * self.M_scaling_list_means[i + 1] * scaling_factor
        FX_mean = (FX_mean_L + FX_mean_R) / 2
        FY_mean = (FY_mean_L + FY_mean_R) / 2
        FZ_mean = (FZ_mean_L + FZ_mean_R) / 2
        MX_mean = (MX_mean_L + MX_mean_R) / 2
        MY_mean = (MY_mean_L + MY_mean_R) / 2
        MZ_mean = (MZ_mean_L + MZ_mean_R) / 2
        FX_0 = 0.0
        FY_0 = 0.0
        FZ_0 = 0.0
        MX_0 = 0.0
        MY_0 = 0.0
        MZ_0 = 0.0
        LP.set_srf_angle(self.srf_angle)
        LP.set_wing_motion_direct(theta_L, eta_L, phi_L, xi_L, theta_R, eta_R, phi_R, xi_R, n_pts)
        LP.set_mean_forces(FX_mean, FY_mean, FZ_mean, MX_mean, MY_mean, MZ_mean)
        LP.set_FT_0(FX_0, FY_0, FZ_0, MX_0, MY_0, MZ_0)
        Fg = np.array([0.0, 0.0, -FZ_0])
        LP.set_Fg(Fg)
        FD = np.array([-FX_0, 0.0, 0.0])
        LP.set_FD(FD)
        LP.compute_tip_forces_custom(wing_length, joint_L, joint_R, LE_pt, TE_pt, m_clr, m_clr, m_clr_2, m_clr_2, Rwing_on, 0.0, plot_wingtip_forces=False, plot_mean_forces=plot_mean_forces, plot_lollipops=plot_lollipops)
        img_width = panel['window_size'][0]
        img_height = panel['window_size'][1]
        p_scale = panel['camera_parallel_scale']
        clip_range = panel['camera_clip']
        cam_pos = panel['camera_position']
        view_up = panel['camera_up']
        forces = '_aero_only_forces'
        wingtip_forces = ''
        if plot_mean_forces == True:
            mean_vector = '_mean_force_vector_'
        else:
            mean_vector = ''
        if plot_lollipops == True:
            lollipop = '_lollipops'
        else:
            lollipop = ''
        test_name = exp_name + forces + wingtip_forces + mean_vector + lollipop
        img_name = destination.name
        LP.take_image(img_width, img_height, p_scale, cam_pos, clip_range, view_up, save_loc, img_name, dpi)
    return LP
