"""Aerodynamic replay calculations.
"""

import math
import pathlib
import numpy as np
import scipy.io


def bilateral_force_traces(model, component):
    """Bilateral sum using the LEFT run's scaling for both wings."""
    Fg = model.mass * model.g
    FgR = model.mass * model.g * model.R_fly
    results = []
    for index in [0, 2]:
        time = np.linspace(0, 1, np.shape(model.FT_SRF_list_means[index][0, :])[0])
        if component < 3:
            scaling = model.F_scaling_list_means[index]
            body_scale = Fg
        else:
            scaling = model.M_scaling_list_means[index]
            body_scale = FgR
        left = model.FT_SRF_list_means[index][component, :] * scaling
        right = model.FT_SRF_list_means[index + 1][component, :] * scaling
        trace = (left + right) / body_scale
        results.append(dict(time=time, trace=trace, mean=np.mean(trace)))
    return results

class RoboAnalysis:

    def __init__(self):
        self.dt = 0.1
        self.dx = 0.1
        self.dy = 0.1
        self.dz = 0.1
        self.g = 9800.0
        self.mass = 1.034e-06
        self.rho_fly = 1.18e-09
        self.rho_robo = 8.8e-07
        self.R_fly = 2.7
        self.R_robo = 250.0
        self.t_FT_f_list_means = []
        self.t_FT_s_list_means = []
        self.FT_gravity_list_means = []
        self.FT_data_f_list_means = []
        self.FT_data_s_list_means = []
        self.FT_filt_f_list_means = []
        self.FT_filt_s_list_means = []
        self.FT_out_list_means = []
        self.FT_raw_f_list_means = []
        self.FT_trace_f_list_means = []
        self.bias_data_wing_list_means = []
        self.bias_wing_list_means = []
        self.filter_gain_f_list_means = []
        self.filter_gain_s_list_means = []
        self.wingkin_f_list_means = []
        self.wingkin_s_list_means = []
        self.t_FT_fast_list_means = []
        self.N_FT_fast_list_means = []
        self.t_FT_slow_list_means = []
        self.N_FT_slow_list_means = []
        self.T_fast_list_means = []
        self.T_slow_list_means = []
        self.FT_wb_mean_scaled_list_means = []
        self.FT_SRF_list_means = []
        self.wingkin_SRF_list_means = []
        self.FT_SRF_list_4 = []
        self.FT_SRF_list_5 = []
        self.FT_SRF_list_6 = []
        self.FT_wb_mean_list_means = []
        self.FT_wing_list_means = []
        self.freq_list_means = []
        self.M_scaling_list_means = []
        self.F_scaling_list_means = []

    def add_freq_and_scaling(self, freq_fly, n_wbs):
        """
        create freq variable per trial (input, robofly) and calc scaling from 
        """
        self.n_fly = freq_fly
        self.f = freq_fly
        self.n_robo = n_wbs / (self.t_FT_fast[-1] * 10 ** (-6))
        self.F_scaling = 1000 * (self.rho_fly * self.n_fly ** 2 * self.R_fly ** 4) / (self.rho_robo * self.n_robo ** 2 * self.R_robo ** 4)
        self.M_scaling = 1000 * (self.rho_fly * self.n_fly ** 2 * self.R_fly ** 5) / (self.rho_robo * self.n_robo ** 2 * self.R_robo ** 5)
        self.freq_list_means.append(self.n_fly)
        self.F_scaling_list_means.append(self.F_scaling)
        self.M_scaling_list_means.append(self.M_scaling)

    def load_mat_file(self, file_loc_in, file_name_in, gain_factor=2.0):
        file_path = pathlib.Path(file_loc_in, file_name_in)
        with file_path.open('rb') as handle:
            self.mat_file = scipy.io.loadmat(handle)
        self.t_FT_f = np.squeeze(self.mat_file['t_FT_f'])
        self.t_FT_s = np.squeeze(self.mat_file['t_FT_s'])
        self.FT_gravity = np.squeeze(self.mat_file['FT_G'])
        self.FT_data_f = np.squeeze(self.mat_file['FT_data_f'])
        self.FT_data_s = np.squeeze(self.mat_file['FT_data_s'])
        self.FT_filt_f = np.squeeze(self.mat_file['FT_filt_f'])
        self.FT_filt_s = np.squeeze(self.mat_file['FT_filt_s'])
        self.FT_out = np.squeeze(self.mat_file['FT_out'])
        self.FT_raw_f = np.squeeze(self.mat_file['FT_raw_f'])
        self.FT_raw_s = np.squeeze(self.mat_file['FT_raw_s'])
        self.FT_trace_f = np.squeeze(self.mat_file['FT_trace_f'])
        self.bias_data_wing = np.squeeze(self.mat_file['bias_data_wing'])
        self.bias_wing = np.squeeze(self.mat_file['bias_wing'])
        self.filter_gain_f = np.squeeze(self.mat_file['filter_gain_f'])
        self.filter_gain_s = np.squeeze(self.mat_file['filter_gain_s'])
        self.wingkin_f = np.squeeze(self.mat_file['wingkin_f'])
        self.wingkin_s = np.squeeze(self.mat_file['wingkin_s'])
        self.t_FT_fast = self.wingkin_f[:, 0]
        self.N_FT_fast = self.wingkin_f.shape[0]
        self.t_FT_slow = self.wingkin_s[:, 0] / 5.0
        self.N_FT_slow = self.wingkin_s.shape[0]
        self.T_fast = self.t_FT_fast / self.t_FT_fast[-1] * 7.0
        self.T_slow = self.t_FT_slow / self.t_FT_slow[-1] * 7.0
        self.FT_wing = self.compute_FT_wing(self.FT_out, gain_factor=gain_factor)

    def compute_FT_wing(self, FT_in, gain_factor=1.0):
        L_cross = np.array([[0.0, -42.0, 0.0], [42, 0.0, 0.0], [0.0, 0.0, 0.0]])
        R_wing1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        R_wing2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        R_mat = np.zeros((6, 6))
        R_mat[0:3, 0:3] = R_wing1
        R_mat[3:6, 0:3] = np.dot(R_wing2, L_cross)
        R_mat[3:6, 3:6] = R_wing2
        FT_neg = FT_in
        FT_neg[1, :] = -FT_neg[1, :]
        FT_neg[3, :] = -FT_neg[3, :]
        FT_neg[5, :] = -FT_neg[5, :]
        FT_w = np.dot(R_mat, FT_neg)
        return FT_w * gain_factor

    def add_data_to_list_means(self):
        self.t_FT_f_list_means.append(self.t_FT_f)
        self.t_FT_s_list_means.append(self.t_FT_s)
        self.FT_gravity_list_means.append(self.FT_gravity)
        self.FT_data_f_list_means.append(self.FT_data_f)
        self.FT_data_s_list_means.append(self.FT_data_s)
        self.FT_filt_f_list_means.append(self.FT_filt_f)
        self.FT_filt_s_list_means.append(self.FT_filt_s)
        self.FT_out_list_means.append(self.FT_out)
        self.FT_raw_f_list_means.append(self.FT_raw_f)
        self.FT_trace_f_list_means.append(self.FT_trace_f)
        self.bias_data_wing_list_means.append(self.bias_data_wing)
        self.bias_wing_list_means.append(self.bias_wing)
        self.filter_gain_f_list_means.append(self.filter_gain_f)
        self.filter_gain_s_list_means.append(self.filter_gain_s)
        self.wingkin_f_list_means.append(self.wingkin_f)
        self.wingkin_s_list_means.append(self.wingkin_s)
        self.t_FT_fast_list_means.append(self.t_FT_fast)
        self.N_FT_fast_list_means.append(self.N_FT_fast)
        self.t_FT_slow_list_means.append(self.t_FT_slow)
        self.N_FT_slow_list_means.append(self.N_FT_slow)
        self.T_fast_list_means.append(self.T_fast)
        self.T_slow_list_means.append(self.T_slow)
        self.FT_SRF_list_means.append(self.FT_SRF)
        self.wingkin_SRF_list_means.append(self.wingkin_SRF)
        self.FT_SRF_list_4.append(self.FT_SRF_4)
        self.FT_SRF_list_5.append(self.FT_SRF_5)
        self.FT_SRF_list_6.append(self.FT_SRF_6)
        self.FT_wb_mean_list_means.append(self.FT_wb_mean)
        self.FT_wb_mean_scaled_list_means.append(self.FT_wb_mean_scaled)
        self.FT_wing_list_means.append(self.FT_wing)

    def set_srf_angle(self, srf_in):
        self.srf_angle = srf_in

    def set_body_cg(self, cg_in):
        self.body_cg = cg_in

    def euler_angle_shift(self, phi, theta, eta, xi, beta, phi_shift):
        phi_n = -phi + phi_shift
        eta_n = -eta
        theta_n = theta
        xi_n = -3.0 * xi
        return (phi_n, theta_n, eta_n, xi_n)

    def euler_angle_shift_shift_eta(self, phi, theta, eta, xi, beta, phi_shift):
        phi_n = -phi + phi_shift
        theta_n = theta
        xi_n = -3.0 * xi
        eta_n = -eta - xi_n / 3
        return (phi_n, theta_n, eta_n, xi_n)

    def convert_to_SRF(self, beta, phi_shift, wing_side='L', shift_eta=True, flip_Rwing_force_signs=True):
        """
        wing_side = 'L' or 'R'
        flip_Rwing_force_signs: if true will flip force sign for asymetric forces Fy, Mx, Mz
        """
        wb_select = (self.T_fast >= 4.0) & (self.T_fast < 5.0)
        wb_select_4 = (self.T_fast >= 3.0) & (self.T_fast < 4.0)
        wb_select_5 = (self.T_fast >= 4.0) & (self.T_fast < 5.0)
        wb_select_6 = (self.T_fast >= 5.0) & (self.T_fast < 6.0)
        self.N_pts = np.sum(wb_select)
        self.dt = 1.0 / (self.N_pts * self.f)[0]
        self.t = np.linspace(0, 1 / self.f, num=self.N_pts)
        wb_456 = np.zeros((self.N_pts, 13, 3))
        wb_456[:, :, 0] = np.pi * (self.wingkin_f[wb_select_4, :] / 180.0)
        wb_456[:, :, 1] = np.pi * (self.wingkin_f[wb_select_5, :] / 180.0)
        wb_456[:, :, 2] = np.pi * (self.wingkin_f[wb_select_6, :] / 180.0)
        wb_mean = np.mean(wb_456, axis=2)
        if wing_side == 'L':
            xi_Lt = wb_mean[:, 5]
            theta_Lt = wb_mean[:, 1]
            eta_Lt = wb_mean[:, 2]
            phi_Lt = wb_mean[:, 3]
            if shift_eta == False:
                self.phi_L, self.theta_L, self.eta_L, self.xi_L = self.euler_angle_shift(phi_Lt, theta_Lt, eta_Lt, xi_Lt, -beta, -phi_shift)
            if shift_eta == True:
                self.phi_L, self.theta_L, self.eta_L, self.xi_L = self.euler_angle_shift_shift_eta(phi_Lt, theta_Lt, eta_Lt, xi_Lt, -beta, -phi_shift)
            cg_cross = np.array([[0, -self.body_cg[2], self.body_cg[1]], [self.body_cg[2], 0, -self.body_cg[0]], [-self.body_cg[1], self.body_cg[0], 0]])
            q_beta = np.array([np.cos(self.srf_angle / 2.0), 0.0, np.sin(self.srf_angle / 2.0), 0.0])
            R_beta = self.comp_R(q_beta)
            R_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            self.FT_SRF = np.zeros((6, self.N_pts))
            self.FT_wb_mean_scaled = np.zeros((6, self.N_pts))
            self.FT_SRF_4 = np.zeros((6, self.N_pts))
            self.FT_SRF_5 = np.zeros((6, self.N_pts))
            self.FT_SRF_6 = np.zeros((6, self.N_pts))
            self.wingkin_SRF = np.zeros((4, self.N_pts))
            FT_456 = np.zeros((6, self.N_pts, 3))
            FT_456[:, :, 0] = self.FT_wing[:, wb_select_4]
            FT_456[:, :, 1] = self.FT_wing[:, wb_select_5]
            FT_456[:, :, 2] = self.FT_wing[:, wb_select_6]
            self.FT_4 = np.zeros((6, self.N_pts))
            self.FT_5 = np.zeros((6, self.N_pts))
            self.FT_6 = np.zeros((6, self.N_pts))
            self.FT_4 = self.FT_wing[:, wb_select_4]
            self.FT_5 = self.FT_wing[:, wb_select_5]
            self.FT_6 = self.FT_wing[:, wb_select_6]
            self.FT_wb_mean = np.mean(FT_456, axis=2)
            self.FT_wb_mean_scaled[0:3, :] = self.FT_wb_mean[0:3, :] * self.F_scaling
            self.FT_wb_mean_scaled[3:, :] = self.FT_wb_mean[3:, :] * self.M_scaling
            for i in range(self.N_pts):
                q_phi_L = np.array([np.cos(self.phi_L[i] / 2.0), np.sin(self.phi_L[i] / 2.0), 0.0, 0.0])
                q_theta_L = np.array([np.cos(-self.theta_L[i] / 2.0), 0.0, 0.0, np.sin(-self.theta_L[i] / 2.0)])
                q_eta_L = np.array([np.cos(self.eta_L[i] / 2.0), 0.0, np.sin(self.eta_L[i] / 2.0), 0.0])
                q_L = self.q_mult(q_phi_L, self.q_mult(q_theta_L, q_eta_L))
                R_L = np.transpose(self.comp_R(q_L))
                R_mat = np.zeros((6, 6))
                R_mat[:3, :3] = np.dot(np.transpose(R_beta), R_L)
                R_mat[3:, 3:] = np.dot(np.transpose(R_beta), R_L)
                FT_i = self.FT_wb_mean[:, i]
                FT_4_i = self.FT_4[:, i]
                FT_5_i = self.FT_5[:, i]
                FT_6_i = self.FT_6[:, i]
                self.FT_SRF[:, i] = np.dot(R_mat, FT_i)
                self.FT_SRF_4[:, i] = np.dot(R_mat, FT_4_i)
                self.FT_SRF_5[:, i] = np.dot(R_mat, FT_5_i)
                self.FT_SRF_6[:, i] = np.dot(R_mat, FT_6_i)
                self.wingkin_SRF[0, i] = self.theta_L[i]
                self.wingkin_SRF[1, i] = self.eta_L[i]
                self.wingkin_SRF[2, i] = self.phi_L[i]
                self.wingkin_SRF[3, i] = self.xi_L[i]
            self.FT_SRF[3:, :] += np.dot(cg_cross, self.FT_SRF[:3, :])
            self.FT_SRF_4[3:, :] += np.dot(cg_cross, self.FT_SRF_4[:3, :])
            self.FT_SRF_5[3:, :] += np.dot(cg_cross, self.FT_SRF_5[:3, :])
            self.FT_SRF_6[3:, :] += np.dot(cg_cross, self.FT_SRF_6[:3, :])
        else:
            xi_Rt = wb_mean[:, 5]
            theta_Rt = wb_mean[:, 1]
            eta_Rt = wb_mean[:, 2]
            phi_Rt = wb_mean[:, 3]
            if shift_eta == False:
                self.phi_R, self.theta_R, self.eta_R, self.xi_R = self.euler_angle_shift(phi_Rt, theta_Rt, eta_Rt, xi_Rt, -beta, -phi_shift)
            if shift_eta == True:
                self.phi_R, self.theta_R, self.eta_R, self.xi_R = self.euler_angle_shift_shift_eta(phi_Rt, theta_Rt, eta_Rt, xi_Rt, -beta, -phi_shift)
            cg_cross = np.array([[0, -self.body_cg[2], self.body_cg[1]], [self.body_cg[2], 0, -self.body_cg[0]], [-self.body_cg[1], self.body_cg[0], 0]])
            q_beta = np.array([np.cos(self.srf_angle / 2.0), 0.0, np.sin(self.srf_angle / 2.0), 0.0])
            R_beta = self.comp_R(q_beta)
            R_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            self.FT_SRF = np.zeros((6, self.N_pts))
            self.FT_wb_mean_scaled = np.zeros((6, self.N_pts))
            self.FT_SRF_4 = np.zeros((6, self.N_pts))
            self.FT_SRF_5 = np.zeros((6, self.N_pts))
            self.FT_SRF_6 = np.zeros((6, self.N_pts))
            self.wingkin_SRF = np.zeros((4, self.N_pts))
            FT_456 = np.zeros((6, self.N_pts, 3))
            FT_456[:, :, 0] = self.FT_wing[:, wb_select_4]
            FT_456[:, :, 1] = self.FT_wing[:, wb_select_5]
            FT_456[:, :, 2] = self.FT_wing[:, wb_select_6]
            self.FT_4 = np.zeros((6, self.N_pts))
            self.FT_5 = np.zeros((6, self.N_pts))
            self.FT_6 = np.zeros((6, self.N_pts))
            self.FT_4 = self.FT_wing[:, wb_select_4]
            self.FT_5 = self.FT_wing[:, wb_select_5]
            self.FT_6 = self.FT_wing[:, wb_select_6]
            self.FT_wb_mean = np.mean(FT_456, axis=2)
            self.FT_wb_mean_scaled[0:3, :] = self.FT_wb_mean[0:3, :] * self.F_scaling
            self.FT_wb_mean_scaled[3:, :] = self.FT_wb_mean[3:, :] * self.M_scaling
            self.FT_wb_mean[1, :] = -self.FT_wb_mean[1, :]
            self.FT_wb_mean[3, :] = -self.FT_wb_mean[3, :]
            self.FT_wb_mean[5, :] = -self.FT_wb_mean[5, :]
            self.FT_wb_mean_scaled[1, :] = -self.FT_wb_mean_scaled[1, :]
            self.FT_wb_mean_scaled[3, :] = -self.FT_wb_mean_scaled[3, :]
            self.FT_wb_mean_scaled[5, :] = -self.FT_wb_mean_scaled[5, :]
            for i in range(self.N_pts):
                if flip_Rwing_force_signs == True:
                    q_phi_R = np.array([np.cos(self.phi_R[i] / 2.0), np.sin(self.phi_R[i] / 2.0), 0.0, 0.0])
                    q_theta_R = np.array([np.cos(-self.theta_R[i] / 2.0), 0.0, 0.0, np.sin(-self.theta_R[i] / 2.0)])
                else:
                    q_phi_R = np.array([np.cos(-self.phi_R[i] / 2.0), np.sin(-self.phi_R[i] / 2.0), 0.0, 0.0])
                    q_theta_R = np.array([np.cos(self.theta_R[i] / 2.0), 0.0, 0.0, np.sin(self.theta_R[i] / 2.0)])
                q_eta_R = np.array([np.cos(self.eta_R[i] / 2.0), 0.0, np.sin(self.eta_R[i] / 2.0), 0.0])
                q_R = self.q_mult(q_phi_R, self.q_mult(q_theta_R, q_eta_R))
                R_R = np.transpose(self.comp_R(q_R))
                R_mat = np.zeros((6, 6))
                R_mat[:3, :3] = np.dot(np.transpose(R_beta), R_R)
                R_mat[3:, 3:] = np.dot(np.transpose(R_beta), R_R)
                FT_i = self.FT_wb_mean[:, i]
                FT_4_i = self.FT_4[:, i]
                FT_5_i = self.FT_5[:, i]
                FT_6_i = self.FT_6[:, i]
                self.FT_SRF[:, i] = np.dot(R_mat, FT_i)
                self.FT_SRF_4[:, i] = np.dot(R_mat, FT_4_i)
                self.FT_SRF_5[:, i] = np.dot(R_mat, FT_5_i)
                self.FT_SRF_6[:, i] = np.dot(R_mat, FT_6_i)
                self.wingkin_SRF[0, i] = self.theta_R[i]
                self.wingkin_SRF[1, i] = self.eta_R[i]
                self.wingkin_SRF[2, i] = self.phi_R[i]
                self.wingkin_SRF[3, i] = self.xi_R[i]
            self.FT_SRF[3:, :] += np.dot(cg_cross, self.FT_SRF[:3, :])
            self.FT_SRF_4[3:, :] += np.dot(cg_cross, self.FT_SRF_4[:3, :])
            self.FT_SRF_5[3:, :] += np.dot(cg_cross, self.FT_SRF_5[:3, :])
            self.FT_SRF_6[3:, :] += np.dot(cg_cross, self.FT_SRF_6[:3, :])
            self.FT_SRF[1, :] = -self.FT_SRF[1, :]
            self.FT_SRF[3, :] = -self.FT_SRF[3, :]
            self.FT_SRF[5, :] = -self.FT_SRF[5, :]
            self.FT_SRF_4[1, :] = -self.FT_SRF_4[1, :]
            self.FT_SRF_4[3, :] = -self.FT_SRF_4[3, :]
            self.FT_SRF_4[5, :] = -self.FT_SRF_4[5, :]
            self.FT_SRF_5[1, :] = -self.FT_SRF_5[1, :]
            self.FT_SRF_5[3, :] = -self.FT_SRF_5[3, :]
            self.FT_SRF_5[5, :] = -self.FT_SRF_5[5, :]
            self.FT_SRF_6[1, :] = -self.FT_SRF_6[1, :]
            self.FT_SRF_6[3, :] = -self.FT_SRF_6[3, :]
            self.FT_SRF_6[5, :] = -self.FT_SRF_6[5, :]

    def q_mult(self, qA, qB):
        QA = np.array([[qA[0], -qA[1], -qA[2], -qA[3]], [qA[1], qA[0], -qA[3], qA[2]], [qA[2], qA[3], qA[0], -qA[1]], [qA[3], -qA[2], qA[1], qA[0]]])
        qC = np.dot(QA, qB)
        qC_norm = math.sqrt(pow(qC[0], 2) + pow(qC[1], 2) + pow(qC[2], 2) + pow(qC[3], 2))
        if qC_norm > 0.01:
            qC /= qC_norm
        else:
            qC = np.array([1.0, 0.0, 0.0, 0.0])
        return qC

    def comp_R(self, q):
        R = np.array([[2 * pow(q[0], 2) - 1 + 2 * pow(q[1], 2), 2 * q[1] * q[2] + 2 * q[0] * q[3], 2 * q[1] * q[3] - 2 * q[0] * q[2]], [2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * pow(q[0], 2) - 1 + 2 * pow(q[2], 2), 2 * q[2] * q[3] + 2 * q[0] * q[1]], [2 * q[1] * q[3] + 2 * q[0] * q[2], 2 * q[2] * q[3] - 2 * q[0] * q[1], 2 * pow(q[0], 2) - 1 + 2 * pow(q[3], 2)]])
        return R
