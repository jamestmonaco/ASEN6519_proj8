# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 08:27:32 2022

@author: gothp
"""

#%% import necessary modules
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt

#%% Loading in the data
data = sci.loadmat("Assignment_8_data.mat")

# And collecting ALL the variables from it
# OL tracking SNR (direct & reflected, L1 & L2) / V/V
OL_L1_d_snr = data['OL_L1_d_snr']
OL_L1_r_snr = data['OL_L1_r_snr']

OL_L2_d_snr = data['OL_L2_d_snr']
OL_L2_r_snr = data['OL_L2_r_snr']
# OL tracking reference carrier phase # m
OL_phi_ref_L1_d = data['OL_phi_ref_L1_d']
OL_phi_ref_L1_r = data['OL_phi_ref_L1_r']

OL_phi_ref_L2_d = data['OL_phi_ref_L2_d']
OL_phi_ref_L2_r = data['OL_phi_ref_L2_r']
# OL tracking excess phase measurement / rad
OL_phi_res_L1_d = data['OL_phi_res_L1_d']
OL_phi_res_L1_r = data['OL_phi_res_L1_r']

OL_phi_res_L2_d = data['OL_phi_res_L2_d']
OL_phi_res_L2_r = data['OL_phi_res_L2_r']

# Receiver position in ECEF coordinates
Rx_X = data['Rx_X'] # m
Rx_Y = data['Rx_Y'] # m
Rx_Z = data['Rx_Z'] # m
Rx_clk_bias = data['Rx_clk_bias'] # Receiver clock bias / s
Rx_timestamp = data['Rx_timestamp'] # Receiver timestamps with clock bias / s
# GPS satellite clock bias at the direct & reflected signal transmission and relativistic corrections / s
gps_clk_bias_d = data['gps_clk_bias_d']
gps_clk_bias_r = data['gps_clk_bias_r']
gps_relsv_d = data['gps_relsv_d']
gps_relsv_r = data['gps_relsv_r']
# GPS satellite ECEF coordinates at the direct & reflected signal transmission / m
gps_pos_d = data['gps_pos_d']
gps_pos_r = data['gps_pos_r']
# Transmission time of direct and reflected signal measurement  / s
gps_time_d = data['gps_time_d']
gps_time_r = data['gps_time_r']
gps_wk = data['gps_wk'] # GPS week
# Elevation, azimuth angles from SP to GPS
sp_az = data['sp_az'] # deg
sp_el = data['sp_el'] # deg
# Geographic coordinates of the SP track
sp_lat = data['sp_lat'] # deg
sp_lon = data['sp_lon'] # deg
sp_mss = data['sp_mss'] # Surface height of the SP track from DTU18 MSS / m
sp_pos = data['sp_pos'] # ECEF coordinates of the SP track / m
sp_tideModel = data['sp_tideModel']

#%% Part 1: 