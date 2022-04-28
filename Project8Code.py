# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 08:27:32 2022

@author: gothp
"""

#%% import necessary modules
import numpy as np
import datetime
import pycircstat as pcs
import scipy.io as sci
import matplotlib.pyplot as plt
from utilities.gpsl1ca import L1CA_CODE_RATE, L1CA_CODE_LENGTH, L1CA_CARRIER_FREQ

# Useful variables 
c = 2.998e8                     # m/s
L1CA_CARRIER_FREQ = 1.57542e9   # hz
L2C_CARRIER_FREQ = 1227.6e6     # hz
L5_CARRIER_FREQ = 1176.45e6     # hz
freqs_all = [L1CA_CARRIER_FREQ, L2C_CARRIER_FREQ, L5_CARRIER_FREQ]
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
Rx_datetime = np.array([datetime.datetime.utcfromtimestamp(Rx_timestamp[t]) for t in range(len(Rx_timestamp))])

# GPS satellite clock bias at the direct & reflected signal transmission and relativistic corrections (s)
gps_clk_bias_d = data['gps_clk_bias_d'][0,:]
gps_clk_bias_r = data['gps_clk_bias_r'][0,:]
gps_relsv_d = data['gps_relsv_d'][0,:]
gps_relsv_r = data['gps_relsv_r'][0,:]

# GPS satellite ECEF coordinates at the direct & reflected signal transmission (m)
gps_pos_d = data['gps_pos_d']
gps_pos_r = data['gps_pos_r']

# Transmission time of direct and reflected signal measurement  (s)
gps_time_d = data['gps_time_d'][0,:]
gps_time_r = data['gps_time_r'][:,0]
gps_wk = data['gps_wk'] # GPS week

# Elevation, azimuth angles from SP to GPS
sp_az = data['sp_az'][0,:] # deg
sp_el = data['sp_el'][0,:] # deg

# Geographic coordinates of the SP track
sp_lat = data['sp_lat'][0,:] # deg
sp_lon = data['sp_lon'][0,:] # deg
sp_mss = data['sp_mss'][0,:] # Surface height of the SP track from DTU18 MSS (m)
sp_pos = data['sp_pos'] # ECEF coordinates of the SP track (m)
sp_tideModel = data['sp_tideModel'][:,0]

# other important data
dt = np.mean(np.diff(Rx_timestamp)) # average timestep (s)
M = np.size(Rx_timestamp)           # number of total points

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

def rx_height(pos_sp, pos_rx, el, epochs):
    '''%--------------------------------
    A function to return the reciever height wrt the specular point

    Parameters
    ----------
    pos_sp : Nx1x3 array
        ECEF position of the specular point at all epochs (m)
    pos_rx : Nx1x3 array
        ECEF position of the reciever at all epochs (m)
    el : 1xN array
        elevation of satellite/rx wrt sp at all epochs (deg)
    epochs : 1xM numpy.array
        the desired epochs (index)

    Returns
    -------
    h : 1xM numpy.array
        the height of the reciever (m)
    '''    
    # cleaning the inputs and isolating the epochs desired
    pos_sp = np.array(pos_sp[:,epochs])
    pos_rx = np.array(pos_rx[:,epochs])
    el = np.deg2rad(np.array(el[epochs])) # assuming spectral reflection
    
    # calculating rx height as the norm between the ECEF rx and sp 
    R = np.linalg.norm([pos_sp - pos_rx], axis=1)[0]
    h = np.sin(el) * R
    return h

# preparing variables needed to calc. Fresnel zone
pos_rx = np.array([Rx_X, Rx_Y, Rx_Z,])
epochs = [0,-1]
rx_h = rx_height(sp_pos, pos_rx, sp_el, epochs) # should be within LEO altitudes 
theta = sp_az[epochs]
e = sp_el[epochs]

# calculating fresnel zone for each carrier.
# Comes out in format: [[a1, a2... ], [b1,b2...], [R1, R2...]] 
fresnel_L1 = FresnelZone(1, e, rx_h, theta)
fresnel_L2 = FresnelZone(2, e, rx_h, theta)
fresnel_L5 = FresnelZone(5, e, rx_h, theta)

#%% Part 2) Coherence detection using reflected signal SNR and carrier phase
# NOTE: From the console, do : `pip install pycircstat`
# Signal SNR: given
# Circular length and circular kurtosis: lecture 23, slide 7
# Circularlength can also be calc. via pycircstat: resultant_vector_length 
# Kurtosis can also be calc via pycircstat: kurtosis

# Signal SNR is given

# setting up for statistical calculation

# calculating kurtosis for each epoch
dphi = np.array( np.deg2rad( np.diff(OL_phi_res_L1_d) ) )
N = 20 # number of samples to perform stats over for each epoch 
circ_length = []
K = []
for i in range(M-1 - N):    # Simply not doing the last N samples, for simplicity
    idx = np.arange(i,(N+i),1)  # indices to sum over
    dphi_i = dphi[idx]          # window of data 
    
    # calculating stats
    circ_length_i = pcs.resultant_vector_length(dphi_i)
    #K_i = pcs.kurtosis(dphi_i) # this doesn't work for some reason
    # (alternatively, from the slides)
    #eta = 1 / N * abs( sum( np.cos(dphi) + np.sin(dphi)) ) 
    K_i = 1 / N * sum( np.cos( 2 * (dphi_i - abs(dphi_i)) ) )
    
    # saving the data
    circ_length.append(circ_length_i)
    K.append(K_i)

#%% Part 3) Altimetry retrieval for a coherent-reflection segment of [150, 300] seconds from the start of the dataset
# a. Unwrap the direct and reflected L1 & L2 signal excess phase measurements OL_phi_res_* 

lambda_L1 = c / freqs_all[0]
lambda_L2 = c / freqs_all[1]
pi2 = 2 * np.pi

L1_dir_unwrap = np.unwrap(OL_phi_res_L1_d)*lambda_L1/pi2
L2_dir_unwrap = np.unwrap(OL_phi_res_L2_d)*lambda_L2/pi2

L1_ref_unwrap = np.unwrap(OL_phi_res_L1_r)*lambda_L1/pi2
L2_ref_unwrap = np.unwrap(OL_phi_res_L2_r)*lambda_L2/pi2

#%% b. Orbit and clock error corrections
# Obtain phase-based range 
L1_range_dir = OL_phi_ref_L1_d + L1_dir_unwrap
L2_range_dir = OL_phi_ref_L2_d + L2_dir_unwrap

L1_range_ref = OL_phi_ref_L1_r + L1_ref_unwrap
L2_range_ref = OL_phi_ref_L2_r + L1_ref_unwrap

# Clock bias correction
L1_range_d_cor = L1_range_dir - c*(Rx_clk_bias + gps_clk_bias_d + gps_relsv_d)
L2_range_d_cor = L2_range_dir - c*(Rx_clk_bias + gps_clk_bias_d + gps_relsv_d)

L1_range_r_cor = L1_range_ref - c*(Rx_clk_bias + gps_clk_bias_r + gps_relsv_r)
L2_range_r_cor = L2_range_ref - c*(Rx_clk_bias + gps_clk_bias_r + gps_relsv_r)

# Plotting original vs excess measurements
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,8))
ax1.set_title('L1 Direct Signal')
ax1.plot(OL_phi_res_L1_d*lambda_L1/pi2,c='red',label='Original Excess Phase')
ax1.plot(L1_range_d_cor,c='blue',label='Estimated Phase')
ax1.legend()

ax2.set_title('L2 Direct Signal')
ax2.plot(OL_phi_res_L2_d*lambda_L2/pi2,c='red',label='Original Excess Phase')
ax2.plot(L2_range_d_cor,c='blue',label='Estimated Phase')
ax2.legend()

ax3.set_title('L1 Reflected Signal')
ax3.plot(OL_phi_res_L1_r*lambda_L1/pi2,c='red',label='Original Excess Phase')
ax3.plot(L1_range_d_cor,c='blue',label='Estimated Phase')
ax3.legend()

ax4.set_title('L2 Reflected Signal')
ax4.plot(OL_phi_res_L2_r*lambda_L2/pi2,c='red',label='Original Excess Phase')
ax4.plot(L2_range_r_cor,c='blue',label='Estimated Phase')
ax4.legend()

fig.tight_layout()

#%% c. Troposphere Correction


#%% d. Check cycle slips and make corrections if needed 


#%% e. Ionosphere Correction
Iono_err_L1 = 
Iono_err_L2 = 

#%% f. Sea surface height anomaly (SSHA) retrieval