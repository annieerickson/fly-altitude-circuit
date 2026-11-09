"""Robofly wing and force rendering used by the Figure 2C workflow.

Run figures/fig2/run.py --panel fig2C from the repository root.
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import time
import os
import math

import cv2

import scipy.special

from .body_class import BodyModel
from .wing_twist_class import WingModel_L
from .wing_twist_class import WingModel_R
from PIL import Image
from ...paths import ensure_output_path


class Lollipop():

    def __init__(self):
        self.N_pol_theta = 20
        self.N_pol_eta = 24
        self.N_pol_phi = 16
        self.N_pol_xi = 20
        self.N_const = 3
        self.grey = (0.5,0.5,0.5)
        self.black = (0.1,0.1,0.1)
        self.red = (1.0,0.0,0.0)
        self.blue = (0.0,0.0,1.0)
        self.n_skip = 35 #30
        

    def Renderer(self):
        self.ren = vtk.vtkRenderer()
        self.ren.SetUseDepthPeeling(True)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

    def ConstructModel(self,body_only):
        if body_only:
            self.body_mdl = BodyModel()
            self.body_mdl.add_actors(self.ren)
        else:
            self.body_mdl = BodyModel()
            self.body_mdl.add_actors(self.ren)
            self.wing_mdl_L = WingModel_L()
            self.wing_mdl_L.add_actors(self.ren)
            self.wing_mdl_R = WingModel_R()
            self.wing_mdl_R.add_actors(self.ren)
        time.sleep(0.001)

    def SetBodyState(self,s_thorax,s_head,s_abdomen):
        # thorax
        self.body_mdl.transform_thorax(s_thorax)
        # head
        self.body_mdl.transform_head(s_head)
        # abdomen
        self.body_mdl.transform_abdomen(s_abdomen)

    def SetBodyScale(self,scale_in):
        self.scale = scale_in
        self.body_mdl.scale_thorax(scale_in[0])
        self.body_mdl.scale_head(scale_in[1])
        self.body_mdl.scale_abdomen(scale_in[2])

    def SetBodyColor(self,clr_in):
        self.body_mdl.set_Color(clr_in)

    def take_image(self,img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,file_name, dpi=300):
       # Add axes:
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(2.0, 2.0, 2.0)
        axes.SetXAxisLabelText('')
        axes.SetYAxisLabelText('')
        axes.SetZAxisLabelText('')
        axes.SetCylinderRadius(0.2)
        axes.SetConeRadius(0.2)
        
        # Set background:
        self.ren.SetBackground(1.0, 1.0, 1.0)
        self.renWin.SetSize(img_width, img_height)
        
        # Get Camera:
        camera = self.ren.GetActiveCamera()
        camera.SetParallelProjection(True)
        camera.SetParallelScale(p_scale)
        camera.SetPosition(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.SetClippingRange(clip_range[0], clip_range[1])
        camera.SetFocalPoint(*getattr(self, 'camera_focal', (0.0, 0.0, 0.0)))
        camera.SetViewUp(view_up[0], view_up[1], view_up[2])
        camera.OrthogonalizeViewUp()
        
        # Multiple renders help ensure everything is ready
        self.renWin.Render()
        self.renWin.Render()
        
        file_path = str(ensure_output_path(save_loc + file_name))
        
        # Diagnostic info
        print(f"Render window type: {self.renWin.GetClassName()}")
        print(f"Number of actors: {self.ren.GetActors().GetNumberOfItems()}")
        print(f"Window size: {self.renWin.GetSize()}")
        
        
        # Raster formats with high quality
        print("Using high-res raster export...")
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.renWin)
        w2i.SetScale(getattr(self, 'capture_scale', 3))
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()
        
        img_i = w2i.GetOutput()
        n_rows, n_cols, _ = img_i.GetDimensions()
        img_sc = img_i.GetPointData().GetScalars()
        np_img = vtk_to_numpy(img_sc)
        np_img = cv2.flip(np_img.reshape(n_cols, n_rows, 3), 0)
        
        from PIL import Image
        pil_img = Image.fromarray(np_img)
        
        if file_path.endswith('.pdf'):
            pil_img.save(str(file_path), 'PDF', resolution=dpi)
        else:
            pil_img.save(str(file_path), dpi=(dpi, dpi))
        
        print(f"Raster saved: {file_path}")


    #will save to jpg via cv2.imwrite, can not change to pdf or specify dpi
    def take_image_original(self,img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,file_name):
        # Add axes:
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(2.0,2.0,2.0)
        axes.SetXAxisLabelText('')
        axes.SetYAxisLabelText('')
        axes.SetZAxisLabelText('')
        axes.SetCylinderRadius(0.2)
        axes.SetConeRadius(0.2)
        #self.ren.AddActor(axes)
        # Set background:
        self.ren.SetBackground(1.0,1.0,1.0)
        self.renWin.SetSize(img_width,img_height)
        # Get Camera:
        camera = self.ren.GetActiveCamera()
        camera.SetParallelProjection(True)
        # Set view:
        camera.SetParallelScale(p_scale)
        camera.SetPosition(cam_pos[0],cam_pos[1],cam_pos[2])
        camera.SetClippingRange(clip_range[0],clip_range[1])
        camera.SetFocalPoint(0.0,0.0,0.0)
        camera.SetViewUp(view_up[0],view_up[1],view_up[2])
        camera.OrthogonalizeViewUp()
        # Render:
        self.renWin.Render()
        #time.sleep(10.0)
        #self.iren.Start()
        
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.renWin)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()
        img_i = w2i.GetOutput()
        n_rows, n_cols, _ = img_i.GetDimensions()
        img_sc = img_i.GetPointData().GetScalars()
        np_img = vtk_to_numpy(img_sc)
        np_img = cv2.flip(np_img.reshape(n_cols,n_rows,3),0)
        cv_img = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
        file_path = str(ensure_output_path(save_loc + file_name))
        cv2.imwrite(str(file_path),cv_img)
        
    def set_body_motion(self,s_in):
        self.s_body = s_in

    def set_srf_angle(self,beta_in):
        self.beta = beta_in

    def set_wing_motion(self,a_theta_L,a_eta_L,a_phi_L,a_xi_L,a_theta_R,a_eta_R,a_phi_R,a_xi_R,n_pts):
        self.N_pts = n_pts
        self.N_wbs = a_theta_L.shape[1]
        # Create tip traces:
        t = np.linspace(0,1,num=self.N_pts)
        X_theta = self.LegendrePolynomials(self.N_pts,self.N_pol_theta,1)
        X_eta     = self.LegendrePolynomials(self.N_pts,self.N_pol_eta,1)
        X_phi     = self.LegendrePolynomials(self.N_pts,self.N_pol_phi,1)
        X_xi     = self.LegendrePolynomials(self.N_pts,self.N_pol_xi,1)

        self.state_mat_L = np.zeros((8,self.N_pts*self.N_wbs))
        self.state_mat_R = np.zeros((8,self.N_pts*self.N_wbs))
        x_L = np.zeros(8)
        x_R = np.zeros(8)
        for i in range(self.N_wbs):
            # Compute wing kinematic angles:
            theta_L     = np.dot(X_theta[:,:,0],a_theta_L[:,i])
            eta_L         = np.dot(X_eta[:,:,0],a_eta_L[:,i])
            phi_L         = np.dot(X_phi[:,:,0],a_phi_L[:,i])
            xi_L         = np.dot(X_xi[:,:,0],a_xi_L[:,i])
            theta_R     = np.dot(X_theta[:,:,0],a_theta_R[:,i])
            eta_R         = np.dot(X_eta[:,:,0],a_eta_R[:,i])
            phi_R         = np.dot(X_phi[:,:,0],a_phi_R[:,i])
            xi_R         = np.dot(X_xi[:,:,0],a_xi_R[:,i])
            for j in range(self.N_pts):
                x_L[0] = phi_L[j]
                x_L[1] = theta_L[j]
                x_L[2] = eta_L[j]
                x_L[3] = -xi_L[j]
                x_L[4] = 0.0
                x_L[5] = 0.6
                x_L[6] = 0.0
                x_R[0] = phi_R[j]
                x_R[1] = theta_R[j]
                x_R[2] = eta_R[j]
                x_R[3] = -xi_R[j]
                x_R[4] = 0.0
                x_R[5] = -0.6
                x_R[6] = 0.0
                self.state_mat_L[:,i*self.N_pts+j] = self.calculate_state_L(x_L)
                self.state_mat_R[:,i*self.N_pts+j] = self.calculate_state_R(x_R)

    def quat_multiply(self,q_A,q_B):
        QA = np.squeeze(np.array([[q_A[0],-q_A[1],-q_A[2],-q_A[3]],
            [q_A[1],q_A[0],-q_A[3],q_A[2]],
            [q_A[2],q_A[3],q_A[0],-q_A[1]],
            [q_A[3],-q_A[2],q_A[1],q_A[0]]]))
        q_C = np.dot(QA,q_B)
        q_C /= math.sqrt(pow(q_C[0],2)+pow(q_C[1],2)+pow(q_C[2],2)+pow(q_C[3],2))
        return q_C

    def quat_mat(self,s_in):
        q0 = np.squeeze(s_in[0])
        q1 = np.squeeze(s_in[1])
        q2 = np.squeeze(s_in[2])
        q3 = np.squeeze(s_in[3])
        tx = np.squeeze(s_in[4])
        ty = np.squeeze(s_in[5])
        tz = np.squeeze(s_in[6])
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, tx],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1, ty],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        return M

    def calculate_state_L(self,x_in):
        # parameters
        phi = x_in[0]
        theta = x_in[1]
        eta = x_in[2]
        xi = x_in[3]
        root_x = x_in[4]
        root_y = x_in[5]
        root_z = x_in[6]
        # convert to quaternions:
        q_start = np.array([np.cos(self.beta/2.0),0.0,np.sin(self.beta/2.0),0.0])
        q_phi = np.array([np.cos(-phi/2.0),np.sin(-phi/2.0),0.0,0.0])
        q_theta = np.array([np.cos(theta/2.0),0.0,0.0,np.sin(theta/2.0)])
        q_eta = np.array([np.cos(-eta/2.0),0.0,np.sin(-eta/2.0),0.0])
        q_L = self.quat_multiply(q_eta,self.quat_multiply(q_theta,self.quat_multiply(q_phi,q_start)))
        # state out:
        state_out = np.zeros(8)
        state_out[0] = q_L[0]
        state_out[1] = q_L[1]
        state_out[2] = q_L[2]
        state_out[3] = q_L[3]
        state_out[4] = root_x
        state_out[5] = root_y
        state_out[6] = root_z
        state_out[7] = xi
        return state_out

    def calculate_state_R(self,x_in):
        # parameters
        phi = x_in[0]
        theta = x_in[1]
        eta = x_in[2]
        xi = x_in[3]
        root_x = x_in[4]
        root_y = x_in[5]
        root_z = x_in[6]
        # convert to quaternions:
        q_start = np.array([np.cos(self.beta/2.0),0.0,np.sin(self.beta/2.0),0.0])
        #q_start = np.array([1.0,0.0,0.0,0.0])
        q_phi = np.array([np.cos(phi/2.0),np.sin(phi/2.0),0.0,0.0])
        q_theta = np.array([np.cos(-theta/2.0),0.0,0.0,np.sin(-theta/2.0)])
        q_eta = np.array([np.cos(-eta/2.0),0.0,np.sin(-eta/2.0),0.0])
        q_R = self.quat_multiply(q_eta,self.quat_multiply(q_theta,self.quat_multiply(q_phi,q_start)))
        # state out:
        state_out = np.zeros(8)
        state_out[0] = q_R[0]
        state_out[1] = q_R[1]
        state_out[2] = q_R[2]
        state_out[3] = q_R[3]
        state_out[4] = root_x
        state_out[5] = root_y
        state_out[6] = root_z
        state_out[7] = xi
        return state_out

    def set_wing_motion_direct(self,theta_L_in,eta_L_in,phi_L_in,xi_L_in,theta_R_in,eta_R_in,phi_R_in,xi_R_in,n_pts):
        self.N_pts = n_pts
        # Wing kinematic angles:
        self.theta_L     = np.squeeze(theta_L_in)
        self.eta_L         = np.squeeze(eta_L_in)-np.squeeze(xi_L_in)/3.0
        self.phi_L         = np.squeeze(phi_L_in)
        self.xi_L         = np.squeeze(xi_L_in)
        self.theta_R     = np.squeeze(theta_R_in)
        self.eta_R         = np.squeeze(eta_R_in)-np.squeeze(xi_R_in)/3.0
        self.phi_R         = np.squeeze(phi_R_in)
        self.xi_R         = np.squeeze(xi_R_in)

    def set_forces_direct(self,FX_L_in,FY_L_in,FZ_L_in,FX_R_in,FY_R_in,FZ_R_in):
        self.FX_L = FX_L_in
        self.FY_L = FY_L_in
        self.FZ_L = FZ_L_in
        self.FX_R = FX_R_in
        self.FY_R = FY_R_in
        self.FZ_R = FZ_R_in

    def set_mean_forces(self,FX_mean,FY_mean,FZ_mean,MX_mean,MY_mean,MZ_mean):
        self.FX_mean = FX_mean
        self.FY_mean = FY_mean
        self.FZ_mean = FZ_mean
        self.MX_mean = MX_mean
        self.MY_mean = MY_mean
        self.MZ_mean = MZ_mean

    def set_FT_0(self,FX_0,FY_0,FZ_0,MX_0,MY_0,MZ_0):
        self.FX_0 = FX_0
        self.FY_0 = FY_0
        self.FZ_0 = FZ_0
        self.MX_0 = MX_0
        self.MY_0 = MY_0
        self.MZ_0 = MZ_0

    def set_Fg(self,FG_in):
        self.FG = FG_in

    def set_FD(self,FD_in):
        self.FD = FD_in

    def compute_tip_trace(self,wing_length,joint_L,joint_R,LE_pt,TE_pt,color_L,color_R,color_L_trace, color_R_trace,right_wing_on):
        wt_L = np.array([0.0,wing_length,0.0])
        wt_R = np.array([0.0,-wing_length,0.0])
        wt_pts_L = np.array([[0.1,wing_length,0.0],[0.0,wing_length,0.0],[-0.1,wing_length,0.0],[-0.2,wing_length,0.0]])
        wt_pts_R = np.array([[0.1,-wing_length,0.0],[0.0,-wing_length,0.0],[-0.1,-wing_length,0.0],[-0.2,-wing_length,0.0]])
        LE_L = np.array([LE_pt,wing_length,0.0])
        LE_R = np.array([LE_pt,-wing_length,0.0])
        TE_L = np.array([TE_pt,wing_length,0.0])
        TE_R = np.array([TE_pt,-wing_length,0.0])
        wt_trace_L = np.zeros((3,self.N_pts))
        wt_trace_R = np.zeros((3,self.N_pts))
        LE_trace_L = np.zeros((3,self.N_pts))
        LE_trace_R = np.zeros((3,self.N_pts))
        TE_trace_L = np.zeros((3,self.N_pts))
        TE_trace_R = np.zeros((3,self.N_pts))
        beta = self.beta
        for i in range(self.N_pts):
            R_90       = np.array([[np.cos(beta),0.0,np.sin(beta)],[0.0,1.0,0.0],[-np.sin(beta),0.0,np.cos(beta)]])
            R_phi_L   = np.array([[1.0,0.0,0.0],[0.0,np.cos(self.phi_L[i]),-np.sin(self.phi_L[i])],[0.0,np.sin(self.phi_L[i]),np.cos(self.phi_L[i])]])
            R_theta_L = np.array([[np.cos(-self.theta_L[i]),-np.sin(-self.theta_L[i]),0.0],[np.sin(-self.theta_L[i]),np.cos(-self.theta_L[i]),0.0],[0.0,0.0,1.0]])
            R_eta_L   = np.array([[np.cos(self.eta_L[i]),0.0,np.sin(self.eta_L[i])],[0.0,1.0,0.0],[-np.sin(self.eta_L[i]),0.0,np.cos(self.eta_L[i])]])
            R_xi_L       = np.array([[np.cos(-self.xi_L[i]/3.0),0.0,np.sin(-self.xi_L[i]/3.0)],[0.0,1.0,0.0],[np.sin(-self.xi_L[i]/3.0),0.0,np.cos(-self.xi_L[i])]])
            R_phi_R   = np.array([[1.0,0.0,0.0],[0.0,np.cos(-self.phi_R[i]),-np.sin(-self.phi_R[i])],[0.0,np.sin(-self.phi_R[i]),np.cos(-self.phi_R[i])]])
            R_theta_R = np.array([[np.cos(self.theta_R[i]),-np.sin(self.theta_R[i]),0.0],[np.sin(self.theta_R[i]),np.cos(self.theta_R[i]),0.0],[0.0,0.0,1.0]])
            R_eta_R   = np.array([[np.cos(self.eta_R[i]),0.0,np.sin(self.eta_R[i])],[0.0,1.0,0.0],[-np.sin(self.eta_R[i]),0.0,np.cos(self.eta_R[i])]])
            R_xi_R       = np.array([[np.cos(-self.xi_R[i]/3.0),0.0,np.sin(-self.xi_R[i]/3.0)],[0.0,1.0,0.0],[np.sin(-self.xi_R[i]/3.0),0.0,np.cos(-self.xi_R[i])]])
            R_L = np.dot(np.dot(np.dot(R_90,R_phi_L),R_theta_L),R_eta_L)
            R_R = np.dot(np.dot(np.dot(R_90,R_phi_R),R_theta_R),R_eta_R)
            wt_trace_L[:,i] = np.dot(R_L,wt_L)+joint_L
            wt_trace_R[:,i] = np.dot(R_R,wt_R)+joint_R
            lollipop_L = np.zeros((4,3))
            lollipop_R = np.zeros((4,3))
            for j in range(4):
                if j<2:
                    R_j_L = R_L
                    R_j_R = R_R
                elif j==2:
                    R_j_L = np.dot(R_L,R_xi_L)
                    R_j_R = np.dot(R_R,R_xi_R)
                elif j==3:
                    R_j_L = np.dot(np.dot(R_L,R_xi_L),R_xi_L)
                    R_j_R = np.dot(np.dot(R_R,R_xi_R),R_xi_R)
                lollipop_L[j,:] = np.dot(R_j_L,wt_pts_L[j,:])+joint_L
                lollipop_R[j,:] = np.dot(R_j_R,wt_pts_R[j,:])+joint_R
            LE_trace_L[:,i] = np.dot(R_L,LE_L)+joint_L
            LE_trace_R[:,i] = np.dot(R_R,LE_R)+joint_R
            TE_trace_L[:,i] = np.dot(R_L,TE_L)+joint_L
            TE_trace_R[:,i] = np.dot(R_R,TE_R)+joint_R
            if i%self.n_skip==0:
                self.lollipop(lollipop_L,color_L_trace)
                if right_wing_on>0:
                    self.lollipop(lollipop_R,color_R_trace)
        self.tip_trace_L(wt_trace_L,color_L_trace)
        if right_wing_on>0:
            self.tip_trace_R(wt_trace_R,color_R_trace)

    def compute_tip_forces(self,wing_length,joint_L,joint_R,LE_pt,TE_pt,color_L,color_R,color_L_trace, color_R_trace, right_wing_on,beta):
        wt_L = np.array([0.0,wing_length,0.0])
        wt_R = np.array([0.0,-wing_length,0.0])
        wt_pts_L = np.array([[0.1,wing_length,0.0],[0.0,wing_length,0.0],[-0.1,wing_length,0.0],[-0.2,wing_length,0.0]])
        wt_pts_R = np.array([[0.1,-wing_length,0.0],[0.0,-wing_length,0.0],[-0.1,-wing_length,0.0],[-0.2,-wing_length,0.0]])
        LE_L = np.array([LE_pt,wing_length,0.0])
        LE_R = np.array([LE_pt,-wing_length,0.0])
        TE_L = np.array([TE_pt,wing_length,0.0])
        TE_R = np.array([TE_pt,-wing_length,0.0])
        wt_trace_L = np.zeros((3,self.N_pts))
        wt_trace_R = np.zeros((3,self.N_pts))
        LE_trace_L = np.zeros((3,self.N_pts))
        LE_trace_R = np.zeros((3,self.N_pts))
        TE_trace_L = np.zeros((3,self.N_pts))
        TE_trace_R = np.zeros((3,self.N_pts))
        FT_list = []
        wt_list = []
        FT_list_R = [] #RW
        wt_list_R = [] #RW
        a_srf = self.beta
        b_srf = self.beta
        R_90 = np.array([[np.cos(a_srf),0.0,np.sin(a_srf)],[0.0,1.0,0.0],[-np.sin(a_srf),0.0,np.cos(a_srf)]])
        R_beta = np.array([[np.cos(b_srf),0.0,np.sin(b_srf)],[0.0,1.0,0.0],[-np.sin(b_srf),0.0,np.cos(b_srf)]])
        for i in range(self.N_pts):
            R_phi_L   = np.array([[1.0,0.0,0.0],[0.0,np.cos(self.phi_L[i]),-np.sin(self.phi_L[i])],[0.0,np.sin(self.phi_L[i]),np.cos(self.phi_L[i])]])
            R_theta_L = np.array([[np.cos(-self.theta_L[i]),-np.sin(-self.theta_L[i]),0.0],[np.sin(-self.theta_L[i]),np.cos(-self.theta_L[i]),0.0],[0.0,0.0,1.0]])
            R_eta_L   = np.array([[np.cos(self.eta_L[i]),0.0,np.sin(self.eta_L[i])],[0.0,1.0,0.0],[-np.sin(self.eta_L[i]),0.0,np.cos(self.eta_L[i])]])
            R_xi_L       = np.array([[np.cos(-self.xi_L[i]/3.0),0.0,np.sin(-self.xi_L[i]/3.0)],[0.0,1.0,0.0],[np.sin(-self.xi_L[i]/3.0),0.0,np.cos(-self.xi_L[i])]])
            R_phi_R   = np.array([[1.0,0.0,0.0],[0.0,np.cos(-self.phi_R[i]),-np.sin(-self.phi_R[i])],[0.0,np.sin(-self.phi_R[i]),np.cos(-self.phi_R[i])]])
            R_theta_R = np.array([[np.cos(self.theta_R[i]),-np.sin(self.theta_R[i]),0.0],[np.sin(self.theta_R[i]),np.cos(self.theta_R[i]),0.0],[0.0,0.0,1.0]])
            R_eta_R   = np.array([[np.cos(self.eta_R[i]),0.0,np.sin(self.eta_R[i])],[0.0,1.0,0.0],[-np.sin(self.eta_R[i]),0.0,np.cos(self.eta_R[i])]])
            R_xi_R       = np.array([[np.cos(-self.xi_R[i]/3.0),0.0,np.sin(-self.xi_R[i]/3.0)],[0.0,1.0,0.0],[np.sin(-self.xi_R[i]/3.0),0.0,np.cos(-self.xi_R[i])]])
            R_L = np.dot(np.dot(np.dot(R_90,R_phi_L),R_theta_L),R_eta_L)
            R_R = np.dot(np.dot(np.dot(R_90,R_phi_R),R_theta_R),R_eta_R)
            R_FT_L = np.dot(np.dot(np.dot(R_90,R_phi_L),R_theta_L),R_eta_L)
            R_FT_R = np.dot(np.dot(np.dot(R_90,R_phi_R),R_theta_R),R_eta_R)
            wt_trace_L[:,i] = np.dot(R_L,wt_L)+joint_L
            wt_trace_R[:,i] = np.dot(R_R,wt_R)+joint_R
            lollipop_L = np.zeros((4,3))
            lollipop_R = np.zeros((4,3))
            for j in range(4):
                if j<2:
                    R_j_L = R_L
                    R_j_R = R_R
                elif j==2:
                    R_j_L = np.dot(R_L,R_xi_L)
                    R_j_R = np.dot(R_R,R_xi_R)
                elif j==3:
                    R_j_L = np.dot(np.dot(R_L,R_xi_L),R_xi_L)
                    R_j_R = np.dot(np.dot(R_R,R_xi_R),R_xi_R)
                lollipop_L[j,:] = np.dot(R_j_L,wt_pts_L[j,:])+joint_L
                lollipop_R[j,:] = np.dot(R_j_R,wt_pts_R[j,:])+joint_R
            LE_trace_L[:,i] = np.dot(R_L,LE_L)+joint_L
            LE_trace_R[:,i] = np.dot(R_R,LE_R)+joint_R
            TE_trace_L[:,i] = np.dot(R_L,TE_L)+joint_L
            TE_trace_R[:,i] = np.dot(R_R,TE_R)+joint_R
            if i%self.n_skip==0:
                self.lollipop(lollipop_L,color_L_trace)
                FT_i = np.array([self.FX_L[i],self.FY_L[i],self.FZ_L[i]])
                FT_beta = np.dot(R_L,FT_i)
                FT_list.append([FT_beta[0],FT_beta[1],FT_beta[2]])
                wt_list.append([wt_trace_L[0,i],wt_trace_L[1,i],wt_trace_L[2,i]])
                if right_wing_on>0:
                    self.lollipop(lollipop_R,color_R_trace)
                    FT_i_R = np.array([self.FX_R[i],self.FY_R[i],self.FZ_R[i]])
                    FT_beta_R = np.dot(R_R,FT_i_R)
                    FT_list_R.append([FT_beta_R[0],FT_beta_R[1],FT_beta_R[2]])
                    wt_list_R.append([wt_trace_R[0,i],wt_trace_R[1,i],wt_trace_R[2,i]])

        self.tip_trace_L(wt_trace_L,color_L_trace)
        FT_L = np.array(FT_list)
        FT_root_L = np.array(wt_list)
        self.ForceGlyphs(FT_root_L,FT_L,color_L)
        
        if right_wing_on>0:
            self.tip_trace_R(wt_trace_R,color_R_trace)
            FT_R = np.array(FT_list_R)
            FT_root_R = np.array(wt_list_R)
            self.ForceGlyphs(FT_root_R,FT_R,color_R)

        FT_mean = np.zeros((2,3))
        FT_mean[0,0] = np.mean(self.FX_mean)
        FT_mean[0,1] = np.mean(self.FY_mean)
        FT_mean[0,2] = np.mean(self.FZ_mean)
        FT_mean[1,0] = np.mean(self.MX_mean)*0.003
        FT_mean[1,1] = np.mean(self.MY_mean)*0.003
        FT_mean[1,2] = np.mean(self.MZ_mean)*0.003
        # compute cp
        cp_mean = self.compute_cp(FT_mean,3.0)
        FT_mean_root = np.array([[cp_mean[0],cp_mean[1]+0.5,cp_mean[2]],[cp_mean[0],cp_mean[1]+0.5,cp_mean[2]]])

        self.MeanGlyph(FT_mean_root,FT_mean,color_L_trace)
        
        FT_0 = np.zeros((2,3))
        FT_0[0,0] = np.mean(self.FX_0)
        FT_0[0,1] = np.mean(self.FY_0)
        FT_0[0,2] = np.mean(self.FZ_0)
        FT_0[1,0] = np.mean(self.MX_0)*0.003
        FT_0[1,1] = np.mean(self.MY_0)*0.003
        FT_0[1,2] = np.mean(self.MZ_0)*0.003
        cp_0 = self.compute_cp(FT_0,3.0)
        FT_0_root = np.array([[cp_0[0],cp_0[1]+0.5,cp_0[2]],[cp_0[0],cp_0[1]+0.5,cp_0[2]]])
        # Gravity vector:
        FT_G = np.zeros((2,3))
        FT_G[0,0] = self.FG[0]
        FT_G[0,1] = self.FG[1]
        FT_G[0,2] = self.FG[2]
        self.MeanGlyph(FT_0_root,FT_G,self.black)   

        # Body drag vector:
        FT_D = np.zeros((2,3))
        FT_D[0,0] = self.FD[0]
        FT_D[0,1] = self.FD[1]
        FT_D[0,2] = self.FD[2]
        self.MeanGlyph(FT_0_root,FT_D,self.blue)
        if right_wing_on>0:
            self.tip_trace_R(wt_trace_R,color_R_trace)
        # Setup joints:
        j_c = np.array([0.0,0.5,0.0])
        j_r = 0.1
        self.joint(j_c,j_r,color_L)
        self.arm(j_c,FT_0_root[0,:],0.015,self.grey)
        self.arm(j_c,FT_mean_root[0,:],0.015,color_L)

    def compute_tip_forces_custom(self,wing_length,joint_L,joint_R,LE_pt,TE_pt,color_L,color_R,color_L_trace, color_R_trace, right_wing_on,beta, plot_wingtip_forces=True, plot_mean_forces=True, plot_lollipops=True):
        #n_skip = 30
        wt_L = np.array([0.0,wing_length,0.0])
        wt_R = np.array([0.0,-wing_length,0.0])
        wt_pts_L = np.array([[0.1,wing_length,0.0],[0.0,wing_length,0.0],[-0.1,wing_length,0.0],[-0.2,wing_length,0.0]])
        wt_pts_R = np.array([[0.1,-wing_length,0.0],[0.0,-wing_length,0.0],[-0.1,-wing_length,0.0],[-0.2,-wing_length,0.0]])
        LE_L = np.array([LE_pt,wing_length,0.0])
        LE_R = np.array([LE_pt,-wing_length,0.0])
        TE_L = np.array([TE_pt,wing_length,0.0])
        TE_R = np.array([TE_pt,-wing_length,0.0])
        wt_trace_L = np.zeros((3,self.N_pts))
        wt_trace_R = np.zeros((3,self.N_pts))
        LE_trace_L = np.zeros((3,self.N_pts))
        LE_trace_R = np.zeros((3,self.N_pts))
        TE_trace_L = np.zeros((3,self.N_pts))
        TE_trace_R = np.zeros((3,self.N_pts))
        if plot_wingtip_forces:
            FT_list = []
            wt_list = []
            FT_list_R = []
            wt_list_R = []
        a_srf = self.beta
        b_srf = self.beta
        R_90 = np.array([[np.cos(a_srf),0.0,np.sin(a_srf)],[0.0,1.0,0.0],[-np.sin(a_srf),0.0,np.cos(a_srf)]])
        R_beta = np.array([[np.cos(b_srf),0.0,np.sin(b_srf)],[0.0,1.0,0.0],[-np.sin(b_srf),0.0,np.cos(b_srf)]])
        for i in range(self.N_pts):
            R_phi_L   = np.array([[1.0,0.0,0.0],[0.0,np.cos(self.phi_L[i]),-np.sin(self.phi_L[i])],[0.0,np.sin(self.phi_L[i]),np.cos(self.phi_L[i])]])
            R_theta_L = np.array([[np.cos(-self.theta_L[i]),-np.sin(-self.theta_L[i]),0.0],[np.sin(-self.theta_L[i]),np.cos(-self.theta_L[i]),0.0],[0.0,0.0,1.0]])
            R_eta_L   = np.array([[np.cos(self.eta_L[i]),0.0,np.sin(self.eta_L[i])],[0.0,1.0,0.0],[-np.sin(self.eta_L[i]),0.0,np.cos(self.eta_L[i])]])
            R_xi_L       = np.array([[np.cos(-self.xi_L[i]/3.0),0.0,np.sin(-self.xi_L[i]/3.0)],[0.0,1.0,0.0],[np.sin(-self.xi_L[i]/3.0),0.0,np.cos(-self.xi_L[i])]])
            R_phi_R   = np.array([[1.0,0.0,0.0],[0.0,np.cos(-self.phi_R[i]),-np.sin(-self.phi_R[i])],[0.0,np.sin(-self.phi_R[i]),np.cos(-self.phi_R[i])]])
            R_theta_R = np.array([[np.cos(self.theta_R[i]),-np.sin(self.theta_R[i]),0.0],[np.sin(self.theta_R[i]),np.cos(self.theta_R[i]),0.0],[0.0,0.0,1.0]])
            R_eta_R   = np.array([[np.cos(self.eta_R[i]),0.0,np.sin(self.eta_R[i])],[0.0,1.0,0.0],[-np.sin(self.eta_R[i]),0.0,np.cos(self.eta_R[i])]])
            R_xi_R       = np.array([[np.cos(-self.xi_R[i]/3.0),0.0,np.sin(-self.xi_R[i]/3.0)],[0.0,1.0,0.0],[np.sin(-self.xi_R[i]/3.0),0.0,np.cos(-self.xi_R[i])]])
            R_L = np.dot(np.dot(np.dot(R_90,R_phi_L),R_theta_L),R_eta_L)
            R_R = np.dot(np.dot(np.dot(R_90,R_phi_R),R_theta_R),R_eta_R)
            R_FT_L = np.dot(np.dot(np.dot(R_90,R_phi_L),R_theta_L),R_eta_L)
            R_FT_R = np.dot(np.dot(np.dot(R_90,R_phi_R),R_theta_R),R_eta_R)
            wt_trace_L[:,i] = np.dot(R_L,wt_L)+joint_L
            wt_trace_R[:,i] = np.dot(R_R,wt_R)+joint_R
            lollipop_L = np.zeros((4,3))
            lollipop_R = np.zeros((4,3))
            for j in range(4):
                if j<2:
                    R_j_L = R_L
                    R_j_R = R_R
                elif j==2:
                    R_j_L = np.dot(R_L,R_xi_L)
                    R_j_R = np.dot(R_R,R_xi_R)
                elif j==3:
                    R_j_L = np.dot(np.dot(R_L,R_xi_L),R_xi_L)
                    R_j_R = np.dot(np.dot(R_R,R_xi_R),R_xi_R)
                lollipop_L[j,:] = np.dot(R_j_L,wt_pts_L[j,:])+joint_L
                lollipop_R[j,:] = np.dot(R_j_R,wt_pts_R[j,:])+joint_R
            LE_trace_L[:,i] = np.dot(R_L,LE_L)+joint_L
            LE_trace_R[:,i] = np.dot(R_R,LE_R)+joint_R
            TE_trace_L[:,i] = np.dot(R_L,TE_L)+joint_L
            TE_trace_R[:,i] = np.dot(R_R,TE_R)+joint_R
            if i%self.n_skip==0:
                if plot_lollipops==True:
                    self.lollipop(lollipop_L,color_L_trace)
                if plot_wingtip_forces:
                    FT_i = np.array([self.FX_L[i],self.FY_L[i],self.FZ_L[i]])
                    FT_beta = np.dot(R_L,FT_i)
                    FT_list.append([FT_beta[0],FT_beta[1],FT_beta[2]])
                    wt_list.append([wt_trace_L[0,i],wt_trace_L[1,i],wt_trace_L[2,i]])
                if right_wing_on>0:
                    if plot_lollipops==True:
                        self.lollipop(lollipop_R,color_R_trace)
                    if plot_wingtip_forces:
                        FT_i_R = np.array([self.FX_R[i],self.FY_R[i],self.FZ_R[i]])
                        FT_beta_R = np.dot(R_R,FT_i_R)
                        FT_list_R.append([FT_beta_R[0],FT_beta_R[1],FT_beta_R[2]])
                        wt_list_R.append([wt_trace_R[0,i],wt_trace_R[1,i],wt_trace_R[2,i]])

        self.tip_trace_L(wt_trace_L,color_L_trace)
        if plot_wingtip_forces:
            FT_L = np.array(FT_list)
            FT_root_L = np.array(wt_list)
            self.ForceGlyphs_tube_radius(FT_root_L,FT_L,color_L)
        
        if right_wing_on>0:
            self.tip_trace_R(wt_trace_R,color_R_trace)
            if plot_wingtip_forces:
                FT_R = np.array(FT_list_R)
                FT_root_R = np.array(wt_list_R)
                self.ForceGlyphs_tube_radius(FT_root_R,FT_R,color_R)

        FT_mean = np.zeros((2,3))
        FT_mean[0,0] = np.mean(self.FX_mean)
        FT_mean[0,1] = np.mean(self.FY_mean)
        FT_mean[0,2] = np.mean(self.FZ_mean)
        FT_mean[1,0] = np.mean(self.MX_mean)*0.003
        FT_mean[1,1] = np.mean(self.MY_mean)*0.003
        FT_mean[1,2] = np.mean(self.MZ_mean)*0.003
        
        # compute cp
        cp_mean = self.compute_cp(FT_mean,3.0)
        FT_mean_root = np.array([[cp_mean[0],cp_mean[1]+0.5,cp_mean[2]],[cp_mean[0],cp_mean[1]+0.5,cp_mean[2]]])

        if plot_mean_forces:
            self.MeanGlyph(FT_mean_root,FT_mean,color_L_trace)
        
        FT_0 = np.zeros((2,3))
        FT_0[0,0] = np.mean(self.FX_0)
        FT_0[0,1] = np.mean(self.FY_0)
        FT_0[0,2] = np.mean(self.FZ_0)
        FT_0[1,0] = np.mean(self.MX_0)*0.003
        FT_0[1,1] = np.mean(self.MY_0)*0.003
        FT_0[1,2] = np.mean(self.MZ_0)*0.003
        cp_0 = self.compute_cp(FT_0,3.0)
        FT_0_root = np.array([[cp_0[0],cp_0[1]+0.5,cp_0[2]],[cp_0[0],cp_0[1]+0.5,cp_0[2]]])
        #print(FT_0)
        #self.MeanGlyph(FT_0_root,FT_0,self.grey)
        # Gravity vector:
        FT_G = np.zeros((2,3))
        FT_G[0,0] = self.FG[0]
        FT_G[0,1] = self.FG[1]
        FT_G[0,2] = self.FG[2]
        self.MeanGlyph(FT_0_root,FT_G,self.black)   

        # Body drag vector:
        FT_D = np.zeros((2,3))
        FT_D[0,0] = self.FD[0]
        FT_D[0,1] = self.FD[1]
        FT_D[0,2] = self.FD[2]
        self.MeanGlyph(FT_0_root,FT_D,self.blue)
        if right_wing_on>0:
            self.tip_trace_R(wt_trace_R,color_R_trace)
        # Setup joints:
        j_c = np.array([0.0,0.5,0.0])
        j_r = 0.1
        self.joint(j_c,j_r,color_L)
        self.arm(j_c,FT_0_root[0,:],0.015,self.grey)
        self.arm(j_c,FT_mean_root[0,:],0.015,color_L)


    def compute_cp(self,FT_in,wing_L):
        #A = np.array([[0.0,FT_in[0,2],-FT_in[0,1]],[-FT_in[0,2],0.0,FT_in[0,0]],[FT_in[0,1],-FT_in[0,0],0.0],[FT_in[0,0],FT_in[0,1],FT_in[0,2]]])
        #b = np.array([[FT_in[1,0]],[FT_in[1,1]],[FT_in[1,2]],[0.0]])
        #A = np.array([[0.0,FT_in[0,2]],[-FT_in[0,2],0.0],[FT_in[0,1],-FT_in[0,0]]])
        #b = np.array([[FT_in[1,0]],[FT_in[1,1]],[FT_in[1,2]]])
        #c = wing_L*np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.dot(np.transpose(A),b))
        cp = np.zeros(3)
        #cp[0] = c[0]
        #cp[1] = c[1]
        #cp[2] = c[2]
        return cp

    def tip_trace_L(self,tip_pts_in,tip_color):
        tip_pts = vtk.vtkPoints()
        tip_pts.SetDataTypeToFloat()
        # add points:
        for i in range(self.N_pts):
            tip_pts.InsertNextPoint(tip_pts_in[0,i],tip_pts_in[1,i],tip_pts_in[2,i])
        # root spline
        tip_spline = vtk.vtkParametricSpline()
        tip_spline.SetPoints(tip_pts)
        tip_function_src = vtk.vtkParametricFunctionSource()
        tip_function_src.SetParametricFunction(tip_spline)
        tip_function_src.SetUResolution(tip_pts.GetNumberOfPoints())
        tip_function_src.Update()
        # Radius interpolation
        tip_radius_interp = vtk.vtkTupleInterpolator()
        tip_radius_interp.SetInterpolationTypeToLinear()
        tip_radius_interp.SetNumberOfComponents(1)
        # Tube radius
        tip_radius = vtk.vtkDoubleArray()
        N_spline = tip_function_src.GetOutput().GetNumberOfPoints()
        tip_radius.SetNumberOfTuples(N_spline)
        tip_radius.SetName("TubeRadius")
        tMin = 0.01
        tMax = 0.01
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            tip_radius.SetTuple1(i, t)
        tubePolyData_tip = vtk.vtkPolyData()
        tubePolyData_tip = tip_function_src.GetOutput()
        tubePolyData_tip.GetPointData().AddArray(tip_radius)
        tubePolyData_tip.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        tuber_tip = vtk.vtkTubeFilter()
        tuber_tip.SetInputData(tubePolyData_tip)
        tuber_tip.SetNumberOfSides(6)
        tuber_tip.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(tubePolyData_tip)
        lineMapper.SetScalarRange(tubePolyData_tip.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(tuber_tip.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_tip_L = vtk.vtkActor()
        self.tubeActor_tip_L.GetProperty().SetColor(tip_color[0],tip_color[1],tip_color[2])
        self.tubeActor_tip_L.SetMapper(tubeMapper)
        self.ren.AddActor(self.tubeActor_tip_L)

    def tip_trace_R(self,tip_pts_in,tip_color):
        tip_pts = vtk.vtkPoints()
        tip_pts.SetDataTypeToFloat()
        # add points:
        for i in range(self.N_pts):
            tip_pts.InsertNextPoint(tip_pts_in[0,i],tip_pts_in[1,i],tip_pts_in[2,i])
        # root spline
        tip_spline = vtk.vtkParametricSpline()
        tip_spline.SetPoints(tip_pts)
        tip_function_src = vtk.vtkParametricFunctionSource()
        tip_function_src.SetParametricFunction(tip_spline)
        tip_function_src.SetUResolution(tip_pts.GetNumberOfPoints())
        tip_function_src.Update()
        # Radius interpolation
        tip_radius_interp = vtk.vtkTupleInterpolator()
        tip_radius_interp.SetInterpolationTypeToLinear()
        tip_radius_interp.SetNumberOfComponents(1)
        # Tube radius
        tip_radius = vtk.vtkDoubleArray()
        N_spline = tip_function_src.GetOutput().GetNumberOfPoints()
        tip_radius.SetNumberOfTuples(N_spline)
        tip_radius.SetName("TubeRadius")
        tMin = 0.01
        tMax = 0.01
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            tip_radius.SetTuple1(i, t)
        tubePolyData_tip = vtk.vtkPolyData()
        tubePolyData_tip = tip_function_src.GetOutput()
        tubePolyData_tip.GetPointData().AddArray(tip_radius)
        tubePolyData_tip.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        tuber_tip = vtk.vtkTubeFilter()
        tuber_tip.SetInputData(tubePolyData_tip)
        tuber_tip.SetNumberOfSides(6)
        tuber_tip.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(tubePolyData_tip)
        lineMapper.SetScalarRange(tubePolyData_tip.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(tuber_tip.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_tip_R = vtk.vtkActor()
        self.tubeActor_tip_R.GetProperty().SetColor(tip_color[0],tip_color[1],tip_color[2])
        self.tubeActor_tip_R.SetMapper(tubeMapper)
        self.ren.AddActor(self.tubeActor_tip_R)

    def remove_tip_traces(self):
        self.ren.RemoveActor(self.tubeActor_tip_L)
        self.ren.RemoveActor(self.tubeActor_tip_R)

    def lollipop(self,tip_pts_in,color_in):
        # sphere:
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(tip_pts_in[0,0],tip_pts_in[0,1],tip_pts_in[0,2])
        sphere.SetRadius(0.02)
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        sphere_actor = vtk.vtkActor()
        sphere_actor.GetProperty().SetColor(color_in[0],color_in[1],color_in[2])
        sphere_actor.SetMapper(sphere_mapper)
        self.ren.AddActor(sphere_actor)
        # tube:
        tip_pts = vtk.vtkPoints()
        tip_pts.SetDataTypeToFloat()
        # add points:
        for i in range(tip_pts_in.shape[0]):
            tip_pts.InsertNextPoint(tip_pts_in[i,0],tip_pts_in[i,1],tip_pts_in[i,2])
        # root spline
        tip_spline = vtk.vtkParametricSpline()
        tip_spline.SetPoints(tip_pts)
        tip_function_src = vtk.vtkParametricFunctionSource()
        tip_function_src.SetParametricFunction(tip_spline)
        tip_function_src.SetUResolution(tip_pts.GetNumberOfPoints())
        tip_function_src.Update()
        # Radius interpolation
        tip_radius_interp = vtk.vtkTupleInterpolator()
        tip_radius_interp.SetInterpolationTypeToLinear()
        tip_radius_interp.SetNumberOfComponents(1)
        # Tube radius
        tip_radius = vtk.vtkDoubleArray()
        N_spline = tip_function_src.GetOutput().GetNumberOfPoints()
        tip_radius.SetNumberOfTuples(N_spline)
        tip_radius.SetName("TubeRadius")
        tMin = 0.01
        tMax = 0.01
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            tip_radius.SetTuple1(i, t)
        tubePolyData_tip = vtk.vtkPolyData()
        tubePolyData_tip = tip_function_src.GetOutput()
        tubePolyData_tip.GetPointData().AddArray(tip_radius)
        tubePolyData_tip.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        tuber_tip = vtk.vtkTubeFilter()
        tuber_tip.SetInputData(tubePolyData_tip)
        tuber_tip.SetNumberOfSides(6)
        tuber_tip.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(tubePolyData_tip)
        lineMapper.SetScalarRange(tubePolyData_tip.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(tuber_tip.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        tubeActor = vtk.vtkActor()
        tubeActor.GetProperty().SetColor(color_in[0],color_in[1],color_in[2])
        tubeActor.SetMapper(tubeMapper)
        self.ren.AddActor(tubeActor)

    def ForceGlyphs(self,force_root,force_vec,glyph_color):
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(16)
        arrow.SetTipLength(0.1)
        arrow.SetTipRadius(0.05)
        arrow.SetShaftRadius(0.02)

        N_glyphs = force_root.shape[0]
        #print('N_glyphs: '+str(N_glyphs))

        roots = vtk.vtkPoints()
        velocity = vtk.vtkDoubleArray()
        velocity.SetNumberOfComponents(3)
        velocity.SetNumberOfTuples(N_glyphs)
        velocity.SetName("velocity")
        magnitude = vtk.vtkDoubleArray()
        magnitude.SetNumberOfValues(N_glyphs)
        magnitude.SetName("magnitude")

        FT_mean = np.zeros(6)

        for i in range(N_glyphs):
            roots.InsertNextPoint(force_root[i,0],force_root[i,1],force_root[i,2])
            force_mag = np.linalg.norm(force_vec[i,:])
            if force_mag>0.01:
                e1 = force_vec[i,0]/force_mag
                e2 = force_vec[i,1]/force_mag
                e3 = force_vec[i,2]/force_mag
                vel = [e1,e2,e3]
                vel_copy = vel.copy()
                velocity.SetTuple(i, vel_copy)
                magnitude.SetValue(i, force_mag)
                #print('e1: '+str(e1)+', e2: '+str(e2)+', e3:'+str(e3))
            else:
                velocity.SetTuple(i, [0.0,0.0,0.0])
                magnitude.SetValue(i, 0.0)
            FT_mean[0] += force_vec[i,0]/(1.0*N_glyphs)
            FT_mean[1] += force_vec[i,1]/(1.0*N_glyphs)
            FT_mean[2] += force_vec[i,2]/(1.0*N_glyphs)
            FT_mean[3] += (force_vec[i,2]*force_root[i,1]-force_vec[i,1]*force_root[i,2])/(1.0*N_glyphs)
            FT_mean[4] += (force_vec[i,0]*force_root[i,2]-force_vec[i,2]*force_root[i,0])/(1.0*N_glyphs)
            FT_mean[5] += (force_vec[i,1]*force_root[i,0]-force_vec[i,0]*force_root[i,1])/(1.0*N_glyphs)
        poly = vtk.vtkPolyData()
        poly.SetPoints(roots)
        poly.GetPointData().AddArray(velocity)
        poly.GetPointData().SetActiveVectors("velocity")
        poly.GetPointData().AddArray(magnitude)
        poly.GetPointData().SetActiveScalars("magnitude")
        # Create glyph.
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(poly)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetScaleFactor(1.05)
        #glyph.OrientOn()
        glyph.SetVectorModeToUseVector()
        #glyph.SetScaleModeToDataScalingOff()
        #glyph.SetColorModeToColorByScalar()
        #glyph.SetColorModeToColorByScale()
        glyph.Update()

        glyphMapper = vtk.vtkPolyDataMapper()
        glyphMapper.ScalarVisibilityOff()
        glyphMapper.SetInputConnection(glyph.GetOutputPort())

        glyphActor = vtk.vtkActor()
        glyphActor.SetMapper(glyphMapper)
        glyphActor.GetProperty().SetColor(glyph_color[0],glyph_color[1],glyph_color[2])

        self.ren.AddActor(glyphActor)

    
   
    def ForceGlyphs_tube_radius(self, force_root, force_vec, glyph_color):
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(16)
        arrow.SetTipLength(0.1)
        arrow.SetTipRadius(0.05)
        arrow.SetShaftRadius(0.02)

        N_glyphs = force_root.shape[0]

        roots = vtk.vtkPoints()
        velocity = vtk.vtkDoubleArray()
        velocity.SetNumberOfComponents(3)
        velocity.SetNumberOfTuples(N_glyphs)
        velocity.SetName("velocity")
        
        magnitude = vtk.vtkDoubleArray()
        magnitude.SetNumberOfValues(N_glyphs)
        magnitude.SetName("magnitude")

        for i in range(N_glyphs):
            roots.InsertNextPoint(force_root[i,0], force_root[i,1], force_root[i,2])
            force_mag = np.linalg.norm(force_vec[i,:])
            if force_mag > 0.01:
                e1, e2, e3 = force_vec[i,:] / force_mag
                velocity.SetTuple(i, [e1, e2, e3])
                magnitude.SetValue(i, force_mag)
            else:
                velocity.SetTuple(i, [0.0, 0.0, 0.0])
                magnitude.SetValue(i, 0.0)

        poly = vtk.vtkPolyData()
        poly.SetPoints(roots)
        poly.GetPointData().AddArray(velocity)
        poly.GetPointData().SetActiveVectors("velocity")
        poly.GetPointData().AddArray(magnitude)
        poly.GetPointData().SetActiveScalars("magnitude")

        # Create glyph
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(poly)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleFactor(1.05)
        glyph.OrientOn()

        # Custom filter to scale only length
        def ScaleLengthOnly(polydata):
            points = polydata.GetPoints()
            scalars = polydata.GetPointData().GetScalars()
            for i in range(points.GetNumberOfPoints()):
                point = list(points.GetPoint(i))
                scale = scalars.GetValue(i // 6)  # Assuming 6 points per arrow
                point[0] *= scale
                points.SetPoint(i, point)

        programmableFilter = vtk.vtkProgrammableFilter()
        programmableFilter.SetInputConnection(glyph.GetOutputPort())
        programmableFilter.SetExecuteMethod(lambda: ScaleLengthOnly(programmableFilter.GetPolyDataInput()))

        glyphMapper = vtk.vtkPolyDataMapper()
        glyphMapper.SetInputConnection(programmableFilter.GetOutputPort())
        glyphMapper.ScalarVisibilityOff()

        glyphActor = vtk.vtkActor()
        glyphActor.SetMapper(glyphMapper)
        glyphActor.GetProperty().SetColor(glyph_color[0], glyph_color[1], glyph_color[2])

        self.ren.AddActor(glyphActor)

    

    def MeanGlyph(self,force_root,force_vec,color_vec):
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(32)
        arrow.SetTipLength(0.1)
        arrow.SetTipRadius(0.05)
        arrow.SetShaftRadius(0.02)

        N_glyphs = 1
        #print('N_glyphs: '+str(N_glyphs))

        roots = vtk.vtkPoints()
        velocity = vtk.vtkDoubleArray()
        velocity.SetNumberOfComponents(3)
        velocity.SetNumberOfTuples(N_glyphs)
        velocity.SetName("velocity")
        magnitude = vtk.vtkDoubleArray()
        magnitude.SetNumberOfValues(N_glyphs)
        magnitude.SetName("magnitude")

        FT_mean = np.zeros(6)

        for i in range(N_glyphs):
            roots.InsertNextPoint(force_root[i,0],force_root[i,1],force_root[i,2])
            force_mag = np.linalg.norm(force_vec[i,:])
            if force_mag>0.01:
                e1 = force_vec[i,0]/force_mag
                e2 = force_vec[i,1]/force_mag
                e3 = force_vec[i,2]/force_mag
                vel = [e1,e2,e3]
                vel_copy = vel.copy()
                velocity.SetTuple(i, vel_copy)
                magnitude.SetValue(i, force_mag)
                #print('e1: '+str(e1)+', e2: '+str(e2)+', e3:'+str(e3))
            else:
                velocity.SetTuple(i, [0.0,0.0,0.0])
                magnitude.SetValue(i, 0.0)
        poly = vtk.vtkPolyData()
        poly.SetPoints(roots)
        poly.GetPointData().AddArray(velocity)
        poly.GetPointData().SetActiveVectors("velocity")
        poly.GetPointData().AddArray(magnitude)
        poly.GetPointData().SetActiveScalars("magnitude")
        # Create glyph.
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(poly)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetScaleFactor(8.0)
        #glyph.OrientOn()
        glyph.SetVectorModeToUseVector()
        #glyph.SetScaleModeToDataScalingOff()
        #glyph.SetColorModeToColorByScalar()
        #glyph.SetColorModeToColorByScale()
        glyph.Update()

        glyphMapper = vtk.vtkPolyDataMapper()
        glyphMapper.ScalarVisibilityOff()
        glyphMapper.SetInputConnection(glyph.GetOutputPort())

        glyphActor = vtk.vtkActor()
        glyphActor.SetMapper(glyphMapper)
        #glyphActor.GetProperty().SetColor(self.glyph_color2[0],self.glyph_color2[1],self.glyph_color2[2])
        glyphActor.GetProperty().SetColor(color_vec[0],color_vec[1],color_vec[2])

        self.ren.AddActor(glyphActor)

    def MeanGlyph_M(self,force_root,force_vec,color_vec):
        #for forces and torques? 
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(32)
        arrow.SetTipLength(0.1)
        arrow.SetTipRadius(0.05)
        arrow.SetShaftRadius(0.02)

        N_glyphs = 2
        #print('N_glyphs: '+str(N_glyphs))

        roots = vtk.vtkPoints()
        velocity = vtk.vtkDoubleArray()
        velocity.SetNumberOfComponents(3)
        velocity.SetNumberOfTuples(N_glyphs)
        velocity.SetName("velocity")
        magnitude = vtk.vtkDoubleArray()
        magnitude.SetNumberOfValues(N_glyphs)
        magnitude.SetName("magnitude")

        FT_mean = np.zeros(6)

        for i in range(N_glyphs):
            roots.InsertNextPoint(force_root[i,0],force_root[i,1],force_root[i,2])
            force_mag = np.linalg.norm(force_vec[i,:])
            if force_mag>0.01:
                e1 = force_vec[i,0]/force_mag
                e2 = force_vec[i,1]/force_mag
                e3 = force_vec[i,2]/force_mag
                vel = [e1,e2,e3]
                vel_copy = vel.copy()
                velocity.SetTuple(i, vel_copy)
                magnitude.SetValue(i, force_mag)
                #print('e1: '+str(e1)+', e2: '+str(e2)+', e3:'+str(e3))
            else:
                velocity.SetTuple(i, [0.0,0.0,0.0])
                magnitude.SetValue(i, 0.0)
        poly = vtk.vtkPolyData()
        poly.SetPoints(roots)
        poly.GetPointData().AddArray(velocity)
        poly.GetPointData().SetActiveVectors("velocity")
        poly.GetPointData().AddArray(magnitude)
        poly.GetPointData().SetActiveScalars("magnitude")
        # Create glyph.
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(poly)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetScaleFactor(8.0)
        #glyph.OrientOn()
        glyph.SetVectorModeToUseVector()
        #glyph.SetScaleModeToDataScalingOff()
        #glyph.SetColorModeToColorByScalar()
        #glyph.SetColorModeToColorByScale()
        glyph.Update()

        glyphMapper = vtk.vtkPolyDataMapper()
        glyphMapper.ScalarVisibilityOff()
        glyphMapper.SetInputConnection(glyph.GetOutputPort())

        glyphActor = vtk.vtkActor()
        glyphActor.SetMapper(glyphMapper)
        #glyphActor.GetProperty().SetColor(self.glyph_color2[0],self.glyph_color2[1],self.glyph_color2[2])
        glyphActor.GetProperty().SetColor(color_vec[0],color_vec[1],color_vec[2])

        self.ren.AddActor(glyphActor)

    def joint(self,joint_center,joint_radius,joint_color):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(joint_center[0],joint_center[1],joint_center[2])
        sphere.SetRadius(joint_radius)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        sphereActor = vtk.vtkActor()
        sphereActor.SetMapper(mapper)
        sphereActor.GetProperty().SetColor(joint_color[0],joint_color[1],joint_color[2])

        self.ren.AddActor(sphereActor)

    def arm(self,pt_a,pt_b,r_ab,color_vec):
        # tube
        arm_pts = vtk.vtkPoints()
        arm_pts.SetDataTypeToFloat()
        arm_pts.InsertNextPoint(pt_a[0],pt_a[1],pt_a[2])
        arm_pts.InsertNextPoint(pt_b[0],pt_b[1],pt_b[2])
        # spline
        arm_spline = vtk.vtkParametricSpline()
        arm_spline.SetPoints(arm_pts)
        arm_function_src = vtk.vtkParametricFunctionSource()
        arm_function_src.SetParametricFunction(arm_spline)
        arm_function_src.SetUResolution(arm_pts.GetNumberOfPoints())
        arm_function_src.Update()
        # Radius interpolation
        arm_radius_interp = vtk.vtkTupleInterpolator()
        arm_radius_interp.SetInterpolationTypeToLinear()
        arm_radius_interp.SetNumberOfComponents(1)
        # Tube radius
        arm_radius = vtk.vtkDoubleArray()
        N_spline = arm_function_src.GetOutput().GetNumberOfPoints()
        arm_radius.SetNumberOfTuples(N_spline)
        arm_radius.SetName("TubeRadius")
        tMin = r_ab
        tMax = r_ab
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            arm_radius.SetTuple1(i, t)
        tubePolyData_arm = vtk.vtkPolyData()
        tubePolyData_arm = arm_function_src.GetOutput()
        tubePolyData_arm.GetPointData().AddArray(arm_radius)
        tubePolyData_arm.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        tuber_arm = vtk.vtkTubeFilter()
        tuber_arm.SetInputData(tubePolyData_arm)
        tuber_arm.SetNumberOfSides(6)
        tuber_arm.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(tubePolyData_arm)
        lineMapper.SetScalarRange(tubePolyData_arm.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(tuber_arm.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        tubeActor = vtk.vtkActor()
        tubeActor.GetProperty().SetColor(color_vec[0],color_vec[1],color_vec[2])
        tubeActor.SetMapper(tubeMapper)
        self.ren.AddActor(tubeActor)

    def LegendrePolynomials(self,N_pts,N_pol,n_deriv):
        L_basis = np.zeros((N_pts,N_pol,n_deriv))
        x_basis = np.linspace(-1.0,1.0,N_pts,endpoint=True)
        for i in range(n_deriv):
            if i==0:
                # Legendre basis:
                for n in range(N_pol):
                    if n==0:
                        L_basis[:,n,i] = 1.0
                    elif n==1:
                        L_basis[:,n,i] = x_basis
                    else:
                        for k in range(n+1):
                            L_basis[:,n,i] += (1.0/np.power(2.0,n))*np.power(scipy.special.binom(n,k),2)*np.multiply(np.power(x_basis-1.0,n-k),np.power(x_basis+1.0,k))
            else:
                # Derivatives:
                for n in range(N_pol):
                    if n>=i:
                        L_basis[:,n,i] = n*L_basis[:,n-1,i-1]+np.multiply(x_basis,L_basis[:,n-1,i])
        return L_basis

    def TemporalBC(self,a_c,N_pol,N_const):
        X_Legendre = self.LegendrePolynomials(100,N_pol,N_const)
        trace = np.dot(X_Legendre[:,:,0],a_c)
        b_L = np.zeros(9)
        b_L[0:4] = trace[-5:-1]
        b_L[4] = 0.5*(trace[0]+trace[-1])
        b_L[5:9] = trace[1:5]
        b_R = np.zeros(9)
        b_R[0:4] = trace[-5:-1]
        b_R[4] = 0.5*(trace[0]+trace[-1])
        b_R[5:9] = trace[1:5]
        c_per = self.LegendreFit(trace,b_L,b_R,N_pol,N_const)
        return c_per

    def plot_body_trace(self,x_in,clr_in):
        n_body_pts = x_in.shape[1]
        body_pts = vtk.vtkPoints()
        body_pts.SetDataTypeToFloat()
        # add points:
        for i in range(n_body_pts):
            body_pts.InsertNextPoint(x_in[0,i],x_in[1,i],x_in[2,i])
        # root spline
        body_spline = vtk.vtkParametricSpline()
        body_spline.SetPoints(body_pts)
        body_function_src = vtk.vtkParametricFunctionSource()
        body_function_src.SetParametricFunction(body_spline)
        body_function_src.SetUResolution(body_pts.GetNumberOfPoints())
        body_function_src.Update()
        # Radius interpolation
        body_radius_interp = vtk.vtkTupleInterpolator()
        body_radius_interp.SetInterpolationTypeToLinear()
        body_radius_interp.SetNumberOfComponents(1)
        # Tube radius
        body_radius = vtk.vtkDoubleArray()
        N_spline = body_function_src.GetOutput().GetNumberOfPoints()
        body_radius.SetNumberOfTuples(N_spline)
        body_radius.SetName("TubeRadius")
        tMin = 0.1
        tMax = 0.1
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            body_radius.SetTuple1(i, t)
        tubePolyData_body = vtk.vtkPolyData()
        tubePolyData_body = body_function_src.GetOutput()
        tubePolyData_body.GetPointData().AddArray(body_radius)
        tubePolyData_body.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        tuber_body = vtk.vtkTubeFilter()
        tuber_body.SetInputData(tubePolyData_body)
        tuber_body.SetNumberOfSides(6)
        tuber_body.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(tubePolyData_body)
        lineMapper.SetScalarRange(tubePolyData_body.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(tuber_body.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_body = vtk.vtkActor()
        self.tubeActor_body.GetProperty().SetColor(clr_in[0],clr_in[1],clr_in[2])
        self.tubeActor_body.SetMapper(tubeMapper)
        self.ren.AddActor(self.tubeActor_body)

    def make_video(self,video_dir,video_file,view_nr,scale_in,snapshot_dir,snapshot_inter,snapshot_name):
        width_img = 1000
        height_img = 800

        self.body_mdl.set_Color([(0.5,0.5,0.5)])
        self.wing_mdl_L.set_Color([(1.0,0.0,0.0),(1.0,0.0,0.0),0.3])
        self.wing_mdl_R.set_Color([(0.0,0.0,1.0),(0.0,0.0,1.0),0.3])
        # Set model in start position:
        s_t = np.array([np.cos((-35.0/180.0)*np.pi/2.0),0.0,np.sin((-35.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0,1.0])
        s_h = np.array([np.cos((-5.0/180.0)*np.pi/2.0),0.0,np.sin((-5.0/180.0)*np.pi/2.0),0.0,0.6*0.9,0.0,0.42*0.9,1.0])
        s_a = np.array([np.cos((-70.0/180.0)*np.pi/2.0),0.0,np.sin((-70.0/180.0)*np.pi/2.0),0.0,0.0,0.0,-0.1*0.9,1.0])
        #s_w_L = np.ones(8)
        #s_w_L[:7] = self.state_mat_L[:,0]
        #s_w_R = np.ones(8)
        #s_w_R[:7] = self.state_mat_R[:,0]

        self.wing_mdl_L.scale_wing(0.9)
        self.wing_mdl_R.scale_wing(0.9)
        self.wing_mdl_L.transform_wing(self.state_mat_L[:,0])
        self.wing_mdl_R.transform_wing(self.state_mat_R[:,0])

        # Add axes:
        #axes = vtk.vtkAxesActor()
        #axes.SetTotalLength(2.0,2.0,2.0)
        #axes.SetXAxisLabelText('')
        #axes.SetYAxisLabelText('')
        #axes.SetZAxisLabelText('')
        #axes.SetConeRadius(0.1)
        #self.ren.AddActor(axes)
        #
        self.ren.SetBackground(1.0,1.0,1.0)
        self.renWin.SetSize(width_img,height_img)

        camera = self.ren.GetActiveCamera()
        camera.SetParallelProjection(True)

        if view_nr==0:
            # frontal 30 deg up
            #camera.SetParallelScale(2.5)
            camera.SetParallelScale(scale_in)
            camera.SetPosition(8.0, 12.0, 3.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(0.0,0.0,1.0)
            camera.OrthogonalizeViewUp()
        elif view_nr==1:
            # rear view
            #camera.SetParallelScale(2.5)
            camera.SetParallelScale(scale_in)
            camera.SetPosition(-12, 0.0, 0.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(0.0,0.0,1.0)
            camera.OrthogonalizeViewUp()
        elif view_nr==2:
            # top view
            #camera.SetParallelScale(3.5)
            camera.SetParallelScale(scale_in)
            camera.SetPosition(0.0, 0.0, 12.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(1.0,0.0,0.0)
            camera.OrthogonalizeViewUp()
        elif view_nr==3:
            # side view
            #camera.SetParallelScale(3.5)
            camera.SetParallelScale(scale_in)
            camera.SetPosition(0.0, 12.0, 0.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(0.0,0.0,1.0)
            camera.OrthogonalizeViewUp()
        elif view_nr==4:
            # front view
            #camera.SetParallelScale(3.5)
            camera.SetParallelScale(scale_in)
            camera.SetPosition(12.0,0.0,0.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(0.0,0.0,1.0)
            camera.OrthogonalizeViewUp()

        #self.iren.Initialize()
        self.renWin.Render()

        time.sleep(1.0)

        size = (width_img,height_img)

        os.chdir(video_dir)
        out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        N_steps = self.state_mat_L.shape[1]

        print(N_steps)

        s_head       = np.zeros(7)
        s_thorax  = np.zeros(7)
        s_abdomen = np.zeros(7)
        x_wing_L  = np.ones(4)
        s_wing_L  = np.zeros(8)
        x_wing_R  = np.ones(4)
        s_wing_R  = np.zeros(8)

        

        os.chdir(snapshot_dir)

        for i in range(N_steps):
            print(i)
            M_body       = self.quat_mat(self.s_body[:,i])
            q_head       = self.quat_multiply(self.s_body[:4,i],s_h[:4])
            #q_head       = self.quat_multiply(s_h[:4],self.s_body[:4,i])
            p_head       = np.dot(M_body,s_h[4:])
            s_head[:4]       = q_head
            s_head[4:]       = p_head[:3]
            q_thorax       = self.quat_multiply(self.s_body[:4,i],s_t[:4])
            #q_thorax       = self.quat_multiply(s_t[:4],self.s_body[:4,i])
            p_thorax       = np.dot(M_body,s_t[4:])
            s_thorax[:4]  = q_thorax
            s_thorax[4:]  = p_thorax[:3]
            q_abdomen       = self.quat_multiply(self.s_body[:4,i],s_a[:4])
            #q_abdomen       = self.quat_multiply(s_a[:4],self.s_body[:4,i])
            p_abdomen       = np.dot(M_body,s_a[4:])
            s_abdomen[:4] = q_abdomen
            s_abdomen[4:] = p_abdomen[:3]
            q_body           = self.s_body[:4,i]
            q_body[1]      = -q_body[1]
            q_body[2]      = -q_body[2]
            q_body[3]      = -q_body[3]
            #q_wing_L       = self.state_mat_L[:4,i]
            q_wing_L       = self.quat_multiply(self.state_mat_L[:4,i],q_body)
            x_wing_L[:3]  = self.state_mat_L[4:7,i]
            p_wing_L       = np.dot(M_body,x_wing_L)
            s_wing_L[:4]  = q_wing_L
            s_wing_L[4:7] = p_wing_L[:3]
            s_wing_L[7]   = self.state_mat_L[7,i]
            #q_wing_R       = self.state_mat_R[:4,i]
            q_wing_R       = self.quat_multiply(self.state_mat_R[:4,i],q_body)
            x_wing_R[:3]  = self.state_mat_R[4:7,i]
            p_wing_R       = np.dot(M_body,x_wing_R)
            s_wing_R[:4]  = q_wing_R
            s_wing_R[4:7] = p_wing_R[:3]
            s_wing_R[7]   = self.state_mat_R[7,i]
            self.body_mdl.transform_head(s_head)
            self.body_mdl.transform_thorax(s_thorax)
            self.body_mdl.transform_abdomen(s_abdomen)
            self.wing_mdl_L.transform_wing(s_wing_L)
            self.wing_mdl_R.transform_wing(s_wing_R)
            time.sleep(0.01)
            self.renWin.Render()
            #Export a single frame
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.renWin)
            w2i.SetInputBufferTypeToRGB()
            w2i.ReadFrontBufferOff()
            w2i.Update()
            img_i = w2i.GetOutput()
            n_rows, n_cols, _ = img_i.GetDimensions()
            img_sc = img_i.GetPointData().GetScalars()
            np_img = vtk_to_numpy(img_sc)
            np_img = cv2.flip(np_img.reshape(n_cols,n_rows,3),0)
            cv_img = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
            out.write(cv_img)
            if i%snapshot_inter==0:
                s_name = snapshot_name+'_'+str(i)+'.jpg'
                cv2.imwrite(s_name,cv_img)
        time.sleep(1.0)
        out.release()

# -------------------------------------------------------------------------------------------------------------------
