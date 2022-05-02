function [Tropo_Delay_az,dry_mf_vmf,wet_mf_vmf,ZHD,ZWD]=tropo_model(GPSweek,GPStime,lat,long,height,el,az);
%Written by Margaret Scott
%Last updated: October 2, 2020
%contact: margaret.scott-1@colorado.edu

%Function uses the latitude, longitude, height, elevation, and
%azimuth to develop a vector output of the Zenith Total Delay (ZTD)
%calculated using Global Pressure and Time 3 (GPT3) fast on a 1x1 grid and Global Mapping
%Function (VMF3).

%This function calls gpt3_1_fast.m, gpt3_1_fast_readGrid.m,
%vmf3_ht.m, saasthyd.m, asknewet.m, and
%gpt3_1.grd.txt


%References for this code:
%All files for GPT3 and VMF3 downloaded from https://vmf.geo.tuwien.ac.at/
%Formulas for ZTD, ZWD, and ZHD from (along with with general reference):
%   "VMF3/GPT3: Refined Discrete and Empirical Troposphere Mapping Functions"
% by D. Landskron, et al. (2017)
%   "Modeling Troposphere Delays for Space Geodectic Techniques" by D.
% Landskron (May 2017-dissertation)
%   "Effects of Atmospheric Azimuthal Asymmetry on the Analysis of Space
% Geodectic Data" by G. Chen & T. Herring (10 Sept 1997; Journal of
% Geophysical Research)

%Inputs: 
%   GPSweek   - scalar
%   GPStime   - scalar
%   lat, long - 1xn vector [deg]
%   height    - 1xn vector [m]
%   el        - 1xn vector [deg] defined as 0 deg = horizon & 90 deg = zenith
%   az        - 1xn vector [deg], only matters for azimuthal asymmetry

%Outputs:
%   Tropo_Delay_az - 1xn vector [m]; 
%                  => zhd*dry_mf +zwd*wet_mf + az_asym
%   dry_mf_vmf     - dry mapping function [unitless]
%   wet_mf_vmf     - wet mapping function [unitless]
%   ZHD            - 1xn, zenith hydrostatic delay [m]
%   ZWD            - 1xn, zenith wet delay [m]
    
[rows,cols]=size(el);
%Note: MJD must be a scalar for gpt3_1_fast.m. The GPT3 model has a 6hr
%resolution, and if you want times greater than 6hrs apart, you must call
%the gpt3_1_fast.m function multiple times
MJD=44244+(GPSweek(1)*7)+ (GPStime(1)/3600/24); %written assuming that GPSweek is a total & already accounts for rollover

radLat=double(lat.*pi/180); %convert to radians
radLong=double(long.*pi/180); %convert to radians

%GPT3 on 1x1 grid
gpt_3_grid=gpt3_1_fast_readGrid;
[p,T,dT,Tm,e_p,ah,aw,la,undu,Gn_h,Ge_h,Gn_w,Ge_w]=gpt3_1_fast(MJD,radLat,radLong,height,0,gpt_3_grid);

%transpose certain GPT3 outputs for further calculations
p=p.';
Tm=Tm.';
e_p=e_p.';
ah=ah.';
aw=aw.';
la=la.';

%Vienna Mapping Function 3, calculate Zenith distance, & calculate azimuthal asym.

%Initialize mapping funciton outputs
dry_mf_vmf=zeros(1,cols); 
wet_mf_vmf=zeros(1, cols);
ZWD = zeros(1,cols);
ZHD = zeros(1,cols);
az_wet = zeros(1,cols);
az_dry= zeros(1,cols);
zd=double((90-el).*pi/180); %Zenith distance, calculated as 90 deg-elevation angle, then converted to radians

for i=1:cols
    
    [dry_mf_vmf(i), wet_mf_vmf(i)]=vmf3_ht(ah(i),aw(i),MJD, radLat(i),radLong(i),height(i),zd(i)); %Vienna Mapping Function
    ZWD(i) = asknewet(e_p(i),Tm(i),la(i));
    ZHD(i) = saasthyd(p(i),radLat(i),height(i));
    
    az_wet(i)=(1/((sind(el(i))*tand(el(i)))+0.0007))*(Gn_w(i)*cosd(az(i))+Ge_w(i)*sind(az(i))); %wet component; coeff 0.0007 from Chen&Herring, 1997
    az_dry(i)=(1/((sind(el(i))*tand(el(i)))+0.0031))*(Gn_h(i)*cosd(az(i)) + Ge_h(i)*sind(az(i))); %dry component; coeff 0.0031 from Chen&Herring, 1997
        
end


Tropo_Delay_az=ZHD.*dry_mf_vmf + ZWD.*wet_mf_vmf + az_wet + az_dry; %Zenith Total Delay






end