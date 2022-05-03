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
gps_time_r = data['gps_time_r'][0,:]
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
def signalStats(res_phase, N=20):
    '''
    Parameters
    ----------
    res_phase : array
        residual phase measurement.
    N : int
        number of samples to perform stats over for each epoch.

    Returns
    -------
    K: list
        kurtosis.
    circ_length: list
        circular length
    '''
    
    # setting up statistics
    dphi = np.array( np.deg2rad( np.diff(res_phase) ) )   # the difference in excess phase
    M = np.size(res_phase)
    
    circ_length = []
    K = []
    for i in range(M-1 - N):        # Simply not doing the last N samples, for simplicity
        idx = np.arange(i,(N+i),1)  # indices to sum over
        dphi_i = dphi[idx]          # window of data to do statistics over
        
        # calculating stats
        circ_length_i = pcs.resultant_vector_length(dphi_i)
        #K_i = pcs.kurtosis(dphi_i) # this doesn't work for some reason
        # (alternatively, from the slides)
        #eta = 1 / N * abs( sum( np.cos(dphi) + np.sin(dphi)) ) 
        K_i = 1 / N * sum( np.cos( 2 * (dphi_i - abs(dphi_i)) ) )
        
        # saving the data
        circ_length.append(circ_length_i)
        K.append(K_i)
    
    return circ_length, K

# calculating the circular statistics for L1, L2, and L5:
N = 20
# direct signals
[circLength_L1_d, K_L1_d] = signalStats(OL_phi_res_L1_d,N)
[circLength_L2_d, K_L2_d] = signalStats(OL_phi_res_L2_d,N)
# reflected signals
[circLength_L1_r, K_L1_r] = signalStats(OL_phi_res_L1_r,N)
[circLength_L2_r, K_L2_r] = signalStats(OL_phi_res_L2_r,N)

# Plotting:
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(20,10))
plt.suptitle("Entire Dataset")

ax1.set_title("Direct Signal SNR")
ax1.plot(OL_L1_d_snr,c='dodgerblue',label='L1')
ax1.plot(OL_L2_d_snr,c='tomato',label='L2')
ax1.set_xticks(np.linspace(0,len(OL_L1_d_snr),6),np.linspace(0,360,6).astype(int))
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("SNR (V/V)")
ax1.legend()
ax1.grid()

ax2.set_title("Direct Signal Circular Length")
ax2.plot(circLength_L1_d,c='dodgerblue',label='L1')
ax2.plot(circLength_L2_d,c='tomato',label='L2')
ax2.set_xticks(np.linspace(0,len(circLength_L1_d),6),np.linspace(0,360,6).astype(int))
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Circular Length (m)")
ax2.legend()
ax2.grid()

ax3.set_title("Direct Signal Circular Kurtosis")
ax3.plot(K_L1_d,c='dodgerblue',label='L1')
ax3.plot(K_L2_d,c='tomato',label='L2')
ax3.set_xticks(np.linspace(0,len(circLength_L1_d),6),np.linspace(0,360,6).astype(int))
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Circular Kurtosis")
ax3.legend()
ax3.grid()

ax4.set_title("Reflect Signal SNR")
ax4.plot(OL_L1_r_snr,c='dodgerblue',label='L1')
ax4.plot(OL_L2_r_snr,c='tomato',label='L2')
ax4.set_xticks(np.linspace(0,len(OL_L1_r_snr),6),np.linspace(0,360,6).astype(int))
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("SNR (V/V)")
ax4.legend()
ax4.grid()

ax5.set_title("Reflected Signal Circular Length")
ax5.plot(circLength_L1_r,c='dodgerblue',label='L1')
ax5.plot(circLength_L2_r,c='tomato',label='L2')
ax5.set_xticks(np.linspace(0,len(circLength_L1_d),6),np.linspace(0,360,6).astype(int))
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Circular Length (m)")
ax5.legend()
ax5.grid()

ax6.set_title("Reflected Signal Circular Kurtosis")
ax6.plot(K_L1_r,c='dodgerblue',label='L1')
ax6.plot(K_L2_r,c='tomato',label='L2')
ax6.set_xticks(np.linspace(0,len(circLength_L1_d),6),np.linspace(0,360,6).astype(int))
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Circular Kurtosis")
ax6.legend()
ax6.grid()

fig.tight_layout()
plt.show()

#%% Part 3) Altimetry retrieval for a coherent-reflection segment of [150, 300] seconds from the start of the dataset
# a. Unwrap the direct and reflected L1 & L2 signal excess phase measurements OL_phi_res_* 

lambda_L1 = c / freqs_all[0]
lambda_L2 = c / freqs_all[1]
pi2 = 2 * np.pi

geo_range_dir = np.linalg.norm([gps_pos_d - pos_rx], axis=1)[0]
geo_range_ref = np.linalg.norm([sp_pos - gps_pos_r], axis=1)[0] + np.linalg.norm([sp_pos - pos_rx], axis=1)[0]


L1_dir_unwrap = np.unwrap(OL_phi_res_L1_d[7550:15100])*lambda_L1/pi2
L2_dir_unwrap = np.unwrap(OL_phi_res_L2_d[7550:15100])*lambda_L2/pi2

L1_ref_unwrap = np.unwrap(OL_phi_res_L1_r[7550:15100])*lambda_L1/pi2
L2_ref_unwrap = np.unwrap(OL_phi_res_L2_r[7550:15100])*lambda_L2/pi2

# Plotting the unwrapped phase
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.set_title("Direct Signal Unwrapped Phase")
ax1.plot(L1_dir_unwrap,c='dodgerblue',label='L1')
ax1.plot(L2_dir_unwrap,c='tomato',label='L2')
ax1.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Phase (m)")
ax1.legend()
ax1.grid()

ax2.set_title("Reflected Signal Unwrapped Phase")
ax2.plot(L1_ref_unwrap,c='dodgerblue',label='L1')
ax2.plot(L2_ref_unwrap,c='tomato',label='L2')
ax2.set_xticks(np.linspace(0,len(L1_ref_unwrap),6),np.linspace(150,300,6).astype(int))
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Phase (m)")
ax2.legend()
ax2.grid()


fig.tight_layout()
plt.show();

#%% b. Orbit and clock error corrections
# Obtain phase-based range 
L1_range_dir = OL_phi_ref_L1_d[7550:15100] + L1_dir_unwrap
L2_range_dir = OL_phi_ref_L2_d[7550:15100] + L2_dir_unwrap

L1_range_ref = OL_phi_ref_L1_r[7550:15100] + L1_ref_unwrap
L2_range_ref = OL_phi_ref_L2_r[7550:15100] + L1_ref_unwrap

# Clock bias correction
L1_range_d_cor = L1_range_dir - c*(Rx_clk_bias[7550:15100] - gps_clk_bias_d[7550:15100] - gps_relsv_d[7550:15100])
L2_range_d_cor = L2_range_dir - c*(Rx_clk_bias[7550:15100] - gps_clk_bias_d[7550:15100] - gps_relsv_d[7550:15100])

L1_range_r_cor = L1_range_ref - c*(Rx_clk_bias[7550:15100] - gps_clk_bias_r[7550:15100] - gps_relsv_r[7550:15100])
L2_range_r_cor = L2_range_ref - c*(Rx_clk_bias[7550:15100] - gps_clk_bias_r[7550:15100] - gps_relsv_r[7550:15100])

L1_phase_est_d = L1_range_d_cor - geo_range_dir[7550:15100]
L1_phase_est_d = L1_phase_est_d - L1_phase_est_d[0]
L2_phase_est_d = L2_range_d_cor - geo_range_dir[7550:15100]
L2_phase_est_d = L2_phase_est_d - L2_phase_est_d[0]

L1_phase_est_r = L1_range_r_cor - geo_range_ref[7550:15100]
L1_phase_est_r = L1_phase_est_r - L1_phase_est_r[0]
L2_phase_est_r = L2_range_r_cor - geo_range_ref[7550:15100]
L2_phase_est_r = L2_phase_est_r - L2_phase_est_r[0]

# Plotting original vs excess measurements
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,8))
ax1.set_title('L1 Direct Signal')
ax1.plot(L1_dir_unwrap,c='red',label='Unwrapped Excess Phase')
ax1.plot(L1_phase_est_d,c='blue',label='Estimated Phase')
ax1.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Phase (m)")
ax1.grid()
ax1.legend()

ax2.set_title('L2 Direct Signal')
ax2.plot(L2_dir_unwrap,c='red',label='Unwrapped Excess Phase')
ax2.plot(L2_phase_est_d,c='blue',label='Estimated Phase')
ax2.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Phase (m)")
ax2.grid()
ax2.legend()

ax3.set_title('L1 Reflected Signal')
ax3.plot(L1_ref_unwrap,c='red',label='Unwrapped Excess Phase')
ax3.plot(L1_phase_est_r,c='blue',label='Estimated Phase')
ax3.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Phase (m)")
ax3.grid()
ax3.legend()

ax4.set_title('L2 Reflected Signal')
ax4.plot(L2_ref_unwrap,c='red',label='Unwrapped Excess Phase')
ax4.plot(L2_phase_est_r,c='blue',label='Estimated Phase')
ax4.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Phase (m)")
ax4.grid()
ax4.legend()

fig.tight_layout()

#%% c. Troposphere Correction
# this uses the MATLAB functions of project 4 to calculate total zeneth delay

tropo_in_dict = {"GPSweek": gps_wk[0][0], 
                 "GPStime": gps_time_r, 
                 "lat": sp_lat, "long": sp_lon, 
                 "height": sp_mss, 
                 "el": sp_el, "az": sp_az}
sci.savemat("tropo_in.mat", tropo_in_dict)

tropo_out = sci.loadmat("tropo_out.mat")

# Removing the tropo delay (twice) from the reflected signal:
tropo_delay = tropo_out['Tropo_Delay_az'][0][7550:15100]
L1_phase_trop_r = L1_phase_est_r - 2*tropo_delay
L2_phase_trop_r = L2_phase_est_r - 2*tropo_delay

L1_phase_trop_r -= L1_phase_trop_r[0]
L2_phase_trop_r -= L2_phase_trop_r[0]

# Plotting the correction and corrected phase:
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(11,5))
ax1.set_title("Troposphere Delay")
ax1.plot(sp_el[7550:15100],tropo_delay,c='violet',linewidth=2.5)
ax1.set_xlabel("SP to GPS Elevation (deg)")
ax1.set_ylabel("Troposphere Delay (m)")
ax1.grid()

ax2.set_title("Reflected Signal Phase (corrected)")
ax2.plot(L1_phase_trop_r,c='dodgerblue',label='L1')
ax2.plot(L2_phase_trop_r,c='tomato',label='L2')
ax2.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Phase (m)")
ax2.grid()
ax2.legend()

fig.tight_layout()
plt.show()

#%% d. Check cycle slips and make corrections if needed 
L1_phase_cyc_r = L1_phase_trop_r
L1_phase_cyc_r[6200:] += lambda_L1
L2_phase_cyc_r = L2_phase_trop_r
L2_phase_cyc_r[6200:] += lambda_L2

L1_phase_trop_r = L1_phase_est_r - 2*tropo_delay
L2_phase_trop_r = L2_phase_est_r - 2*tropo_delay
L1_phase_trop_r -= L1_phase_trop_r[0]
L2_phase_trop_r -= L2_phase_trop_r[0]

# Plot the comparison:
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(11,5))
ax1.set_title("L1 Phase Estimate")
ax1.plot(L1_phase_trop_r,c='navy',label='Before Correction')
ax1.plot(L1_phase_cyc_r,c='dodgerblue',label='After Correction',alpha=0.75)
ax1.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Phase (m)")
ax1.grid()
ax1.legend()
ax1.axvline(6200,c='black',linewidth=1,linestyle=":")

ax2.set_title("L2 Phase Estimate")
ax2.plot(L2_phase_trop_r,c='chocolate',label='Before Correction')
ax2.plot(L2_phase_cyc_r,c='tomato',label='After Correction',alpha=0.75)
ax2.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Phase (m)")
ax2.grid()
ax2.legend()
ax2.axvline(6200,c='black',linewidth=1,linestyle=":")
   
fig.tight_layout()
plt.show()
 
#%% e. Ionosphere Correction
# Calcualte the TEC:
beta = (1/40.3) * (freqs_all[0]**2 * freqs_all[1] **2)/(freqs_all[1]**2 - freqs_all[0]**2)
STEC_dir = beta * (L1_range_d_cor - L2_range_d_cor) # electrons / m^2
STEC_ref = beta * (L1_range_r_cor - L2_range_r_cor) # electrons / m^2
# Use the TEC to calculate the phase advance:
L1_phase_adv_dir = -(40.3*STEC_dir)/(freqs_all[0]**2)
L2_phase_adv_dir = -(40.3*STEC_dir)/(freqs_all[1]**2)

L1_phase_adv_ref = -(40.3*STEC_ref)/(freqs_all[0]**2)
L2_phase_adv_ref = -(40.3*STEC_ref)/(freqs_all[1]**2)
# Correct the phase values using the phase advance:
L1_phase_cor_d = L1_phase_est_d + L1_phase_adv_dir # Note these are added since the arrays are negative
L1_phase_cor_d -= L1_phase_cor_d[0]
L2_phase_cor_d = L2_phase_est_d + L2_phase_adv_dir ##
L2_phase_cor_d -= L2_phase_cor_d[0]

L1_phase_cor_r = L1_phase_cyc_r + L1_phase_adv_ref ###
L1_phase_cor_r -= L1_phase_cor_r[0]
L2_phase_cor_r = L2_phase_cyc_r + L2_phase_adv_ref ####
L2_phase_cor_r -= L2_phase_cor_r[0]

# Make plots of the delay and corrections:
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,8))
ax1.set_title("Direct Signal Ionospheric Correction")
ax1.plot(L1_phase_adv_dir,c='dodgerblue',label='L1')
ax1.plot(L2_phase_adv_dir,c='tomato',label='L2')
ax1.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Phase Advance (m)")
ax1.grid()
ax1.legend()

ax2.set_title("Direct Signal Phase (corrected)")
ax2.plot(L1_phase_cor_d,c='dodgerblue',label='L1')
ax2.plot(L2_phase_cor_d,c='tomato',label='L2')
ax2.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Phase Advance (m)")
ax2.grid()
ax2.legend()

ax3.set_title("Reflected Signal Ionospheric Correction")
ax3.plot(L1_phase_adv_ref,c='dodgerblue',label='L1')
ax3.plot(L2_phase_adv_ref,c='tomato',label='L2')
ax3.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Phase (m)")
ax3.grid()
ax3.legend()

ax4.set_title("Reflected Signal Phase (corrected)")
ax4.plot(L1_phase_cor_r,c='dodgerblue',label='L1')
ax4.plot(L2_phase_cor_r,c='tomato',label='L2')
ax4.set_xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Phase (m)")
ax4.grid()
ax4.legend()

fig.tight_layout()
plt.show()    

#%% f. Sea surface height anomaly (SSHA) retrieval
# subtract the direct signal excess phase from the reflected signal excess phase
L1_phase_adv_dir -= L1_phase_adv_dir[0]
L2_phase_adv_dir -= L2_phase_adv_dir[0]

L1_phase_cor_r -= L1_phase_cor_r[0]
L2_phase_cor_r -= L2_phase_cor_r[0]

L1_SSHA = (L1_phase_adv_dir - L1_phase_cor_r) / 2 / np.sin(np.radians(sp_el[7550:15100]))
L2_SSHA = (L2_phase_adv_dir - L2_phase_cor_r) / 2 / np.sin(np.radians(sp_el[7550:15100]))

L1_SSHA -= np.mean(L1_SSHA)
L2_SSHA -= np.mean(L2_SSHA)

# Plot the results:
plt.figure(figsize=(8,5))
plt.title("Sea Surface Height Anomaly")
plt.plot(L1_SSHA,c='dodgerblue',label='L1')
plt.plot(L2_SSHA,c='tomato',label='L2')
plt.xticks(np.linspace(0,len(L1_dir_unwrap),6),np.linspace(150,300,6).astype(int))
plt.xlabel("Time (s)")
plt.ylabel("SSHA (m)")
plt.grid()
plt.legend()
