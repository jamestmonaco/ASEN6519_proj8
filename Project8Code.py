# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 08:27:32 2022

@author: gothp
"""

#%% import necessary modules
import numpy as np
import datetime
import pycircstat
import scipy.io as sci
import matplotlib.pyplot as plt
from utilities.gpsl1ca import L1CA_CODE_RATE, L1CA_CODE_LENGTH, L1CA_CARRIER_FREQ

#%% Loading in the data and collecting ALL the variables from it

data = sci.loadmat("Assignment_8_data.mat")

# OL tracking SNR (direct & reflected, L1 & L2) (V/V)
OL_L1_d_snr = data['OL_L1_d_snr'][:,0]
OL_L1_r_snr = data['OL_L1_r_snr'][:,0]
OL_L2_d_snr = data['OL_L2_d_snr'][:,0]
OL_L2_r_snr = data['OL_L2_r_snr'][:,0]

# OL tracking reference carrier phase (m)
OL_phi_ref_L1_d = data['OL_phi_ref_L1_d'][:,0]
OL_phi_ref_L1_r = data['OL_phi_ref_L1_r'][:,0]
OL_phi_ref_L2_d = data['OL_phi_ref_L2_d'][:,0]
OL_phi_ref_L2_r = data['OL_phi_ref_L2_r'][:,0]

# OL tracking excess phase measurement (rad)
OL_phi_res_L1_d = data['OL_phi_res_L1_d'][:,0]
OL_phi_res_L1_r = data['OL_phi_res_L1_r'][:,0]
OL_phi_res_L2_d = data['OL_phi_res_L2_d'][:,0]
OL_phi_res_L2_r = data['OL_phi_res_L2_r'][:,0]

# Receiver position in ECEF coordinates
Rx_X = data['Rx_X'][:,0] # m
Rx_Y = data['Rx_Y'][:,0] # m
Rx_Z = data['Rx_Z'][:,0] # m

Rx_clk_bias = data['Rx_clk_bias'][:,0]      # Receiver clock bias (s)
Rx_timestamp = data['Rx_timestamp'][:,0]    # Receiver timestamps with clock bias (s)

# GPS satellite clock bias at the direct & reflected signal transmission and relativistic corrections (s)
gps_clk_bias_d = data['gps_clk_bias_d'][:,0]
gps_clk_bias_r = data['gps_clk_bias_r'][:,0]
gps_relsv_d = data['gps_relsv_d'][:,0]
gps_relsv_r = data['gps_relsv_r'][:,0]

# GPS satellite ECEF coordinates at the direct & reflected signal transmission (m)
gps_pos_d = data['gps_pos_d'][:,0]
gps_pos_r = data['gps_pos_r'][:,0]

# Transmission time of direct and reflected signal measurement  (s)
gps_time_d = data['gps_time_d'][:,0]
gps_time_r = data['gps_time_r'][:,0]
gps_wk = data['gps_wk'] # GPS week

# Elevation, azimuth angles from SP to GPS
sp_az = data['sp_az'][:,0] # deg
sp_el = data['sp_el'][:,0] # deg

# Geographic coordinates of the SP track
sp_lat = data['sp_lat'][:,0] # deg
sp_lon = data['sp_lon'][:,0] # deg
sp_mss = data['sp_mss'][:,0] # Surface height of the SP track from DTU18 MSS (m)
sp_pos = data['sp_pos'][:,0] # ECEF coordinates of the SP track (m)
sp_tideModel = data['sp_tideModel'][:,0]

#%% Part 1) Calculate the size of the first Fresnel zone for the first and last epoch of signal reflection
# Writing in Kristine Larson's FresnelZone.m function in Python
def FresnelZone(freq, e, h, theta):
    ''' %------------------------------------------------------------------
        % function [firstF] = FreneslZone(freq, e, h, theta)
        %   This function gets the size and center of the First Fresnel Zone ellipse
        %     at the selected  L-band frequency  (freq)
        %     for an Antenna height (h) above the flat reflecting surface
        %     for a satellite elevation angle (e) and azimuth direction (theta)
        %
        %      (this code assumes a horizontal, untilted reflecting surface)   
        %-------------------------------------------------------------------
        % input
        %       freq:  1 2 or 5 :  for L-band frequency (L1,L2, or L5)      
        %       e:  elevation angle in degrees
        %       h: antenna height in meters, above the flat reflecting surface
        %       theta: azimuth angle in degrees 
        %
        % output
        %      firstF: [a, b, R ] in meters where:
        %              a is the semi-major axis, aligned with the satellite azimuth 
        %              b is the semi-minor axis
        %              R locates the center of the ellispe 
        %                on the satellite azimuth direction (theta)
        %                and R meters away from the base of the Antenna.
        %     
        %     The ellipse is located on a flat horizontal surface h meters below
        %     the receiver.                  
        %------------------------------------------------------------------------
    '''
    CLIGHT = 299792458                                        # speed of light, m/sec
    FREQ = np.array([1575.42e6, 1227.6e6, 0, 0, 1176.45e6])   # GPS frequencies, Hz
                                                              # [ L1 L2  0 0 L5]
    CYCLE = CLIGHT/FREQ                                       # wavelength per cycle (m/cycle)
    RAD2M = CYCLE/2/np.pi                                        # (m)
        
    # ------------------
    # delta = locus of points corresponding to a fixed delay;
    # typically the first Fresnel zone is is the 
    # "zone for which the differential phase change across
    # the surface is constrained to lambda/2" (i.e. 1/2 the wavelength)
    delta = CYCLE[freq-1]/2 	# [meters]
    
    # semi-major and semi-minor dimension
    # from the appendix of Larson and Nievinski, 2013
    sin_elev = np.sin(np.radians(e))
    d = delta 
    B = np.sqrt( (2 * d * h / sin_elev) + (d / sin_elev)**2 )  # [meters]
    A = B / sin_elev                                       # [meters]
    
    # determine distance to ellipse center 
    center = (h + delta/np.sin(np.radians(e)))/np.tan(np.radians(e))   	# [meters]
    
    return [A,B,center]

# Need to calculate the height of the reflector to get signal height above it
# calculate antenna height by converting the antenna's ECEF position to range wrt SP ECEF position. 
# The reflection is specular, so elevation of GPS wrt SP == elevation of antenna wrt SP
# with range and elevation, height can be calculated. 

for i in 

#%% Part 2) Coherence detection using reflected signal SNR and carrier phase
### NOTE: From the console, do : `pip install pycircstat`


#%% Part 3) Altimetry retrieval for a coherent-reflection segment of [150, 300] seconds from the start of the dataset
# a. Unwrap the direct and reflected L1 & L2 signal excess phase measurements OL_phi_res_* 


#%% b. Orbit and clock error corrections


#%% c. Troposphere Correction


#%% d. Check cycle slips and make corrections if needed 


#%% e. Ionosphere Correction


#%% f. Sea surface height anomaly (SSHA) retrieval