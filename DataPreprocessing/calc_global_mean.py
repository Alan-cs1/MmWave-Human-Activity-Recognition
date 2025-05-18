# Calculate global mean in Python using matplotlib

# to make plot interactive
#%matplotlib

# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import matplotlib.animation as animation
from IPython import display
from sklearn.cluster import KMeans

from sklearn import metrics
from sklearn.cluster import DBSCAN

# import the pandas module
import pandas as pd

from pykalman import KalmanFilter

# Local Imports
from parseFrame import *
from graphUtilities import *

hz = 18       # sampling rate (FPS)
bb_lx = -4.0
bb_rx = 4.0
bb_ny = -5.0  # depth
bb_fy = 5.0
bb_bz = 0.0   # height
bb_tz = 3.0
bb_vl = -2.0  # velocity
bb_vu = 2.0


n_max_clip = 4

# calculate global mean of the feature
x_sum_glb = 0
y_sum_glb = 0
z_sum_glb = 0
v_sum_glb = 0

x_sum2_glb = 0
y_sum2_glb = 0
z_sum2_glb = 0
v_sum2_glb = 0

numPoints_glb = 0

#'''
x_mean_glb = -0.14
y_mean_glb = 1.65
z_mean_glb = 1.2
v_mean_glb = 0.3

x_std_glb = 0.53
y_std_glb = 0.5
z_std_glb = 0.47
v_std_glb = 0.27
#'''
'''
x_mean_glb = -0.14
y_mean_glb = 1.78
z_mean_glb = 1.3
v_mean_glb = -0.01

x_std_glb = 0.55
y_std_glb = 0.55
z_std_glb = 0.48
v_std_glb = 0.42
'''

# normalize the feature
xs = []
ys = []
zs = []
vs = []
cs = []

ff = []
xx = []
yy = []
zz = []
vv = []


bin_idx = 6
#'''
bin_path = [ '07_22_2023_15_15_52', # walking  0
             '07_22_2023_15_20_34', # walking
             '07_22_2023_15_25_50', # walking
             '07_22_2023_15_30_10', # sitting
             '07_22_2023_15_34_08', # sitting
             '07_22_2023_15_38_31', # sitting  5
             '07_22_2023_15_44_08', # lying
             '07_22_2023_15_51_10', # lying
             '07_22_2023_15_56_26', # lying
             '07_22_2023_16_01_24', # falling
             '07_22_2023_16_06_23', # falling  10
             '07_22_2023_16_11_23', # falling
             '07_22_2023_16_15_49', # headache
             '07_22_2023_16_19_46', # headache
             '07_22_2023_16_23_28', # headache
             '07_22_2023_16_27_49', # sos      15
             '07_22_2023_16_33_18', # sos
             '07_22_2023_16_37_33', # sos
             '08_02_2023_13_35_59', # walking
             '08_02_2023_13_40_07', # sitting  
             '08_02_2023_13_46_25', # sitting  20
             '08_02_2023_13_53_11', # lying
             '08_02_2023_13_59_55', # lying
             '08_02_2023_14_07_32', # lying
             '08_02_2023_14_18_21', # falling  
             '08_02_2023_14_26_51', # falling  25
             '08_02_2023_14_35_26', # falling
             '08_02_2023_14_48_28', # headache
             '08_02_2023_14_56_33', # headache
             '08_02_2023_15_02_35', # sos      
             '08_13_2023_10_30_05', # sitting (0)  30
             '08_13_2023_10_39_46', # sitting (0) 
             '08_13_2023_10_45_42', # sitting (270)
             '08_13_2023_10_50_30', # sitting (270) 
             '08_13_2023_10_54_39', # sitting (180)
             '08_13_2023_10_59_00', # sitting (180) 35
             '08_13_2023_11_04_16', # sitting (90)
             '08_13_2023_11_10_17', # lying (270)
             '08_13_2023_11_15_32', # lying (270) 
             '08_13_2023_11_21_12', # lying (90)
             '08_13_2023_13_42_30', # sitting (270) 40
             '08_13_2023_13_47_48', # sitting (270)
             '08_13_2023_13_51_35', # sitting (90)
             '08_13_2023_13_55_05', # sitting (90)  
             '08_13_2023_13_59_54', # sitting (0, L)
             '08_13_2023_14_02_32', # sitting (0, R)   45
             '08_13_2023_14_05_35', # sitting (270, L)
             '08_23_2023_15_07_09', # walking
             '08_23_2023_15_16_10', # walking
             '08_23_2023_15_28_47', # sitting
             '08_23_2023_15_36_06', # sitting          50
             '08_23_2023_15_42_11', # sitting (270, L)
             '08_23_2023_15_48_08', # sitting
             '08_23_2023_15_53_46', # lying
             '08_23_2023_16_00_19', # lying
             '08_23_2023_16_06_07', # lying (270)      55
             '08_23_2023_16_17_39', # headache
             '08_23_2023_16_23_05'] # sos      
#'''

skip_frame = 25

sensorHeight = 2.0
azi_idx = 0
elevTilt = 15
#aziTilt = [0, 45, 90, 135, 180, 225, 270, 315, 360]
aziTilt = [0, 45, 180, 315]

saveName = ''



def calcChannelSum(frameNum, b_i, azi_idx):
    global numPoints_glb
    global x_sum_glb, y_sum_glb, z_sum_glb, v_sum_glb
    global x_sum2_glb, y_sum2_glb, z_sum2_glb, v_sum2_glb
    global aziTilt

    numPoints = 0
    retValue = 1

    filename = './binData/' + bin_path[b_i] + '/pHistBytes_' + str(frameNum) + '.bin'

    try:
        dfile = open(filename, 'rb', 0)
        #print('open ', filename)
    except:
        print('cannot open ', filename)
        return 0
    #endtry

    frameData = bytes(list(dfile.read()))
    if (frameData):
        # skip the non-action frame
        if frameNum <= skip_frame:
            return retValue

        outputDict = parseStandardFrame(frameData)

        # Number of Points
        if ('numDetectedPoints' in outputDict):
            numPoints = outputDict['numDetectedPoints']
        else:
            numPoints = 0
        #endif

        if (numPoints):
            # Point Cloud
            xs_sum = []
            ys_sum = []
            zs_sum = []
            vs_sum = []

            if ('pointCloud' in outputDict):
                pointCloud = outputDict['pointCloud']
                if (pointCloud is not None):
                    for i in range(numPoints):
                        #x, y, z, Dopper, SNR, Noise
                        #print(i, pointCloud[i,0], pointCloud[i,1], pointCloud[i,2], pointCloud[i,3], pointCloud[i,4], pointCloud[i,5])

                        # Rotate point cloud and tracks to account for elevation and azimuth tilt
                        rotX, rotY, rotZ = eulerRot (pointCloud[i,0], pointCloud[i,1], pointCloud[i,2], elevTilt, aziTilt[azi_idx])
                        pointCloud[i,0] = rotX
                        pointCloud[i,1] = rotY
                        pointCloud[i,2] = rotZ

                        # +sensorHeight
                        pointCloud[i,2] += sensorHeight

                        # clip to visualizaion bounding box
                        pointCloud[i,0] = np.clip(pointCloud[i,0],bb_lx,bb_rx)
                        pointCloud[i,1] = np.clip(pointCloud[i,1],bb_ny,bb_fy)
                        pointCloud[i,2] = np.clip(pointCloud[i,2],bb_bz,bb_tz)

                        doppler = pointCloud[i,3]

                        # normalize snr to 0~1
                        #snr = pointCloud[i,4]
                        #if (snr < SNR_EXPECTED_MIN) or (snr > SNR_EXPECTED_MAX):
                        #    snr = 1.0;
                        #else:
                        #    snr = (snr - SNR_EXPECTED_MIN) / SNR_EXPECTED_RANGE

                        xs_sum.append(pointCloud[i,0])
                        ys_sum.append(pointCloud[i,1])
                        zs_sum.append(pointCloud[i,2])
                        vs_sum.append(doppler)
                    #endfor

                    #print('frame ', outputDict['frameNum'], '[', numPoints, ']', filename)
                #endif
            #endif

            numPoints_glb += numPoints

            # calculate the mean of current frame
            x_sum_glb += sum(xs_sum)
            y_sum_glb += sum(ys_sum)
            z_sum_glb += sum(zs_sum)
            v_sum_glb += sum(vs_sum)

            x_sum2_glb += sum(i*i for i in xs_sum)
            y_sum2_glb += sum(i*i for i in ys_sum)
            z_sum2_glb += sum(i*i for i in zs_sum)
            v_sum2_glb += sum(i*i for i in vs_sum)
        #else:
        #    print('frame ', outputDict['frameNum'], '[', numPoints, 0, ']', filename)
        #endif
    else:
        print('read bin file NG', filename)
        retValue = 0
    #endif

    return retValue
#enddef calcChannelSum()


def calcGlobalMeanStd():
    global numPoints_glb
    global x_sum_glb, y_sum_glb, z_sum_glb, v_sum_glb
    global x_sum2_glb, y_sum2_glb, z_sum2_glb, v_sum2_glb
    global x_mean_glb, y_mean_glb, z_mean_glb, v_mean_glb
    global x_std_glb, y_std_glb, z_std_glb, v_std_glb


    x_sum_glb = 0
    y_sum_glb = 0
    z_sum_glb = 0
    v_sum_glb = 0

    x_sum2_glb = 0
    y_sum2_glb = 0
    z_sum2_glb = 0
    v_sum2_glb = 0

    numPoints_glb = 0

    # calculate global sum of the feature
    # all tracks
    b_i = 0
    while b_i < len(bin_path):
        ret = 1
        frameNum = 1
        while (ret == 1):
            for j in range(len(aziTilt)):
                ret = calcChannelSum(frameNum, b_i, j)
            #endfor
            frameNum += 1
        #endwhile

        b_i += 1
    #endwhile

    # calculate global mean of the feature
    x_mean_glb = x_sum_glb / numPoints_glb
    y_mean_glb = y_sum_glb / numPoints_glb
    z_mean_glb = z_sum_glb / numPoints_glb
    v_mean_glb = v_sum_glb / numPoints_glb

    # std = sqrt(E[X^2] - (E[X])^2)
    x_std_glb = 1
    y_std_glb = 1
    z_std_glb = 1
    v_std_glb = 1

    if x_mean_glb != 0.0:
        x_std_glb = round((x_sum2_glb / numPoints_glb - x_mean_glb**2),4)**0.5
    if y_mean_glb != 0.0:
        y_std_glb = round((y_sum2_glb / numPoints_glb - y_mean_glb**2),4)**0.5
    if z_mean_glb != 0.0:
        z_std_glb = round((z_sum2_glb / numPoints_glb - z_mean_glb**2),4)**0.5
    if v_mean_glb != 0.0:
        v_std_glb = round((v_sum2_glb / numPoints_glb - v_mean_glb**2),4)**0.5

    print(numPoints_glb)
    print('glb_mean: ', round(x_mean_glb,2), round(y_mean_glb,2), round(z_mean_glb,2), round(v_mean_glb,2))
    print('glb_std: ', round(x_std_glb,2), round(y_std_glb,2), round(z_std_glb,2), round(v_std_glb,2))
#enddef calcGlobalMeanStd()



# main

# calculate global mean and std
print('calc global mean and std')
calcGlobalMeanStd()

print('glb_mean: ', round(x_mean_glb,2), round(y_mean_glb,2), round(z_mean_glb,2), round(v_mean_glb,2))
print('glb_std:  ', round(x_std_glb,2), round(y_std_glb,2), round(z_std_glb,2), round(v_std_glb,2))


