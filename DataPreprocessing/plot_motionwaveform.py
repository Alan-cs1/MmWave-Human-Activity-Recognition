# 1D motion waveform in Python using matplotlib

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
bb_lv = -2.0  # velocity
bb_uv = 2.0


n_max_clip = 5


# global mean & std of the features
'''
x_mean_glb = -0.14
y_mean_glb = 1.65
z_mean_glb = 1.2
v_mean_glb = 0.3

x_std_glb = 0.53
y_std_glb = 0.5
z_std_glb = 0.47
v_std_glb = 0.27
'''

#'''
x_mean_glb = 0.0
y_mean_glb = 0.0
z_mean_glb = 1.2
v_mean_glb = 0.3

x_std_glb = 1.42
y_std_glb = 1.42
z_std_glb = 0.47
v_std_glb = 0.27
#'''

# normalize the feature
xs = []
ys = []
zs = []
vs = []
cs = []
ss = []

ff = []
xx = []
yy = []
zz = []
vv = []


bin_idx = 0
bin_path = [ '07_22_2023_15_15_52', # walking          0
             '07_22_2023_15_20_34', # walking
             '07_22_2023_15_25_50', # walking
             '07_22_2023_15_30_10', # sitting
             '07_22_2023_15_34_08', # sitting
             '07_22_2023_15_38_31', # sitting          5
             '07_22_2023_15_44_08', # lying
             '07_22_2023_15_51_10', # lying
             '07_22_2023_15_56_26', # lying
             '07_22_2023_16_01_24', # falling
             '07_22_2023_16_06_23', # falling         10
             '07_22_2023_16_11_23', # falling
             '07_22_2023_16_15_49', # headache
             '07_22_2023_16_19_46', # headache
             '07_22_2023_16_23_28', # headache
             '07_22_2023_16_27_49', # sos             15
             '07_22_2023_16_33_18', # sos
             '07_22_2023_16_37_33', # sos
             '08_02_2023_13_35_59', # walking
             '08_02_2023_13_40_07', # sitting
             '08_02_2023_13_46_25', # sitting         20
             '08_02_2023_13_53_11', # lying
             '08_02_2023_13_59_55', # lying
             '08_02_2023_14_07_32', # lying
             '08_02_2023_14_18_21', # falling
             '08_02_2023_14_26_51', # falling         25
             '08_02_2023_14_35_26', # falling
             '08_02_2023_14_48_28', # headache
             '08_02_2023_14_56_33', # headache
             '08_02_2023_15_02_35', # sos
             '08_13_2023_10_30_05', # sitting (0)     30
             '08_13_2023_10_39_46', # sitting (0)
             '08_13_2023_10_45_42', # sitting (270)
             '08_13_2023_10_50_30', # sitting (270)
             '08_13_2023_10_54_39', # sitting (180)
             '08_13_2023_10_59_00', # sitting (180)   35
             '08_13_2023_11_04_16', # sitting (90)
             '08_13_2023_11_10_17', # lying (270)
             '08_13_2023_11_15_32', # lying (270)
             '08_13_2023_11_21_12', # lying (90)
             '08_13_2023_13_42_30', # sitting (270)   40
             '08_13_2023_13_47_48', # sitting (270)
             '08_13_2023_13_51_35', # sitting (90)
             '08_13_2023_13_55_05', # sitting (90)
             '08_13_2023_13_59_54', # sitting (0, L)
             '08_13_2023_14_02_32', # sitting (0, R)  45
             '08_23_2023_15_07_09', # walking
             '08_23_2023_15_16_10', # walking
             '08_23_2023_15_28_47', # sitting
             '08_23_2023_15_36_06', # sitting
             '08_23_2023_15_42_11', # sitting         50
             '08_23_2023_15_48_08', # sitting
             '08_23_2023_15_53_46', # lying
             '08_23_2023_16_00_19', # lying
             '08_23_2023_16_06_07', # lying (270)
             '08_23_2023_16_17_39', # headache        55
             '08_23_2023_16_23_05'] # sos

skip_frame = 25

sensorHeight = 2.0
azi_idx = 0
elevTilt = 15
aziTilt = [0, 45, 180, 315]
persistent = 5
last_numPoints = [0] * 20
bKF = 1             # default 1
t_cov = 0.005
em_iter = 1
g_cluster_num = 4   # 
cluster_func = 3    # 2: DBSCAN, 3: Geometry 
normalizeFunc = 4   # 

g_filterFunc = 1
g_sort_by_x = 0
g_all_track = 0
g_cluster_in_norm = 0

saveName = ''


def KmeanCluster(xs:list, ys:list, zs:list, vs:list):
    global g_cluster_num

    if len(xs) > 0:
        # establish the model
        pointcloud = np.array([xs, ys, zs, vs]).T

        if (len(pointcloud) < 50):
            Kmean = KMeans(n_clusters=1, n_init = 3, max_iter = 5)
        else:
            Kmean = KMeans(n_clusters=g_cluster_num, n_init = 3, max_iter = 5)
        #endif

        # fit the data
        Kmean.fit(pointcloud)
        c_labels = Kmean.predict(pointcloud)  # labels of each point

        # cluster centers
        c_centers = Kmean.cluster_centers_

        return c_labels, len(c_centers)
    else:
        return [], 0
    #endif

#enddef KmeanCluster()


def DbscanCluster(xs:list, ys:list, zs:list, vs:list):
    global g_cluster_num

    if len(xs) > 0:
        """
        DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        finds core samples in regions of high density and expands clusters from them.
        This algorithm is good for data which contains clusters of similar density.
        """
        # establish the model
        pointcloud = np.array([xs, ys, zs, vs]).T
        #pointcloud = np.array([xs, ys, zs]).T

        sample_num = 60

        db = DBSCAN(eps=0.3, min_samples=sample_num).fit(pointcloud)

        c_labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
        #n_noise = list(c_labels).count(-1)

        return c_labels, n_clusters
    else:
        return [], 0
    #endif

#enddef DbscanCluster()


def GeometryCluster(xs:list, ys:list, zs:list, vs:list):
    global g_cluster_num
    
    c_labels = []
    n_clusters = g_cluster_num
    
    # calculate the center point
    xc, yc, zc, vc = filterFunc(xs, ys, zs, vs)
    
    #print(xc, yc, zc, vc)

    x_threshold = xc
    y_threshold = yc                  
    z_threshold = zc  # 0.9
    
    if len(zs) > 0:
        for i in range(len(zs)):
            if n_clusters == 2:           
                # 2 group clustering
                if zs[i] >= z_threshold:
                    c_labels.append(0)
                else:
                    c_labels.append(1)   
                #endif
            elif n_clusters == 4:  
                # 4 group clustering
                if zs[i] >= z_threshold:
                    if xs[i] >= x_threshold:
                        c_labels.append(0)
                    else: 
                        c_labels.append(1)
                else:
                    if xs[i] >= x_threshold:
                        c_labels.append(2)
                    else:
                        c_labels.append(3)      
                #endif            
            elif n_clusters == 8:  
                # 8 group clustering
                if zs[i] >= z_threshold:
                    if ys[i] >= y_threshold:
                        if xs[i] >= x_threshold:
                            c_labels.append(0)
                        else:
                            c_labels.append(1)
                    else:
                        if xs[i] >= x_threshold:
                            c_labels.append(2)
                        else:
                            c_labels.append(3)
                else:
                    if ys[i] >= y_threshold:
                        if xs[i] >= x_threshold:
                            c_labels.append(4)
                        else:
                            c_labels.append(5)
                    else:
                        if xs[i] >= x_threshold:
                            c_labels.append(6)
                        else:
                            c_labels.append(7)    
                #endif        
            else: 
                c_labels.append(0)
        #endffor

        '''
        print(len(zs), len(c_labels), xc, yc, zc, vc, 
              list(c_labels).count(0), list(c_labels).count(1), list(c_labels).count(2), list(c_labels).count(3))
        print(len(zs), len(c_labels), xc, yc, zc, vc, 
              list(c_labels).count(0), list(c_labels).count(1), list(c_labels).count(2), list(c_labels).count(3),
              list(c_labels).count(4), list(c_labels).count(5), list(c_labels).count(6), list(c_labels).count(7))
        '''
        #print(c_labels)
        
        return c_labels, n_clusters
    else:
        return [], 0
    #endif   
    
#enddef GeometryCluster()



def KalmanF(xx:list):
    global t_cov
    global em_iter
    global bKF

    if bKF == 1:
        kf_x = KalmanFilter(transition_matrices = [1], # default = state plus noise
                observation_matrices = [1],            # default = state plus noise
                initial_state_mean = 0,                # default = zero
                initial_state_covariance = 1,          # default = identify
                observation_covariance = 1,            # default = identify, max_error = 3
                transition_covariance = t_cov)         # default = identify

        kf_x = kf_x.em(xx, n_iter = em_iter)
        (xx_tmp, covs_x) = kf_x.smooth(xx)

        # col_0
        return np.round(xx_tmp[:,0],2)
    else:
        return xx
    #endif
#enddef KalmanF()


def filterFunc(tmp_xs:list, tmp_ys:list, tmp_zs:list, tmp_vs:list):
    global g_filterFunc

    x = 0
    y = 0
    z = 0
    v = 0

    if g_filterFunc == 1:
        # mean
        if len(tmp_xs) > 0:
            x = mean(tmp_xs)
            y = mean(tmp_ys)
            z = mean(tmp_zs)
            v = mean(tmp_vs)
        #endif
    elif g_filterFunc == 2:
        # median
        if len(tmp_xs) != 0:
            x = median(tmp_xs)
        if len(tmp_ys) != 0:
            y = median(tmp_ys)
        if len(tmp_zs) != 0:
            z = median(tmp_zs)
        if len(tmp_vs) != 0:
            v = median(tmp_vs)
    elif g_filterFunc == 3:
        # max
        if len(tmp_xs) != 0:
            x = max(tmp_xs)
        if len(tmp_ys) != 0:
            y = max(tmp_ys)
        if len(tmp_zs) != 0:
            z = max(tmp_zs)
        if len(tmp_vs) != 0:
            v = max(tmp_vs)
    elif g_filterFunc == 4:
        # min
        if len(tmp_xs) != 0:
            x = min(tmp_xs)
        if len(tmp_ys) != 0:
            y = min(tmp_ys)
        if len(tmp_zs) != 0:
            z = min(tmp_zs)
        if len(tmp_vs) != 0:
            v = min(tmp_vs)
    else:
        if len(xs_tmp) > 0:
            x = mean(tmp_xs)
            y = mean(tmp_ys)
            z = mean(tmp_zs)
            v = mean(tmp_vs)
        #endif
    #endif

    x = np.clip(x,-n_max_clip,n_max_clip)
    y = np.clip(y,-n_max_clip,n_max_clip)
    z = np.clip(z,-n_max_clip,n_max_clip)
    v = np.clip(v,-n_max_clip,n_max_clip)

    x = round(x,2)
    y = round(y,2)
    z = round(z,2)
    v = round(v,2)

    return x, y, z, v
#enddef filterFunc()


def genMotionWaveform(frameNum):
    global xs, ys, zs, vs, cs, ss
    global ff, xx, yy, zz, vv
    global persistent
    global last_numPoints
    global x_mean_glb, y_mean_glb, z_mean_glb, v_mean_glb
    global x_std_glb, y_std_glb, z_std_glb, v_std_glb
    global normalizeFunc
    global g_sort_by_x
    global bb_lx, bb_rx, bb_ny, bb_fy, bb_bz, bb_tz, bb_uv, bb_lv
    global azi_idx, aziTilt

    numPoints = 0
    retValue = 1

    filename = './binData/' + bin_path[bin_idx] + '/pHistBytes_' + str(frameNum) + '.bin'

    try:
        dfile = open(filename, 'rb', 0)
        #print('open ', filename)
    except:
        print('cannot open ', filename)
        return 0
    #endtry

    frameData = bytes(list(dfile.read()))
    if (frameData):
        outputDict = parseStandardFrame(frameData)

        # Number of Points
        if ('numDetectedPoints' in outputDict):
            numPoints = outputDict['numDetectedPoints']
        else:
            numPoints = 0
        #endif

        # remove the oldest frame from persistent window
        xs[:last_numPoints[frameNum%persistent]] = []
        ys[:last_numPoints[frameNum%persistent]] = []
        zs[:last_numPoints[frameNum%persistent]] = []
        vs[:last_numPoints[frameNum%persistent]] = []
        cs[:last_numPoints[frameNum%persistent]] = []
        ss[:last_numPoints[frameNum%persistent]] = []

        numPoints_new = 0

        if (numPoints):
            # Point Cloud
            if ('pointCloud' in outputDict):
                pointCloud = outputDict['pointCloud']
                if (pointCloud is not None):
                    for i in range(numPoints):
                        #x, y, z, Dopper, SNR, Noise
                        #print(i, pointCloud[i,0], pointCloud[i,1], pointCloud[i,2], pointCloud[i,3], pointCloud[i,4], pointCloud[i,5])

                        # clip to visualizaion bounding box
                        pointCloud[i,0] = np.clip(pointCloud[i,0],bb_lx,bb_rx)
                        pointCloud[i,1] = np.clip(pointCloud[i,1],bb_ny,bb_fy)

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
                        snr = pointCloud[i,4]
                        #if (snr < SNR_EXPECTED_MIN) or (snr > SNR_EXPECTED_MAX):
                        #    snr = 1.0
                        #else:
                        #    snr = (snr - SNR_EXPECTED_MIN) / SNR_EXPECTED_RANGE
                        ##endif

                        ptsize = 0
                        if snr > 0:
                            ptsize = int(math.log2(snr)) + 1
                        #endif

                        # skip point size small than 3 by experimental result
                        if ptsize < 3:
                            continue
                        #endif

                        # persistent window
                        xs.append(pointCloud[i,0])
                        ys.append(pointCloud[i,1])
                        zs.append(pointCloud[i,2])
                        vs.append(doppler)
                        ss.append(ptsize)

                        if abs(doppler) >= 0.1:
                            cs.append([1, 0, 0, 1])
                        else:
                            cs.append([0, 1, 0, 1])
                        #endif

                        numPoints_new += 1
                    #endfor
                #endif
            #endif
        #endif

        last_numPoints[frameNum % persistent] = numPoints_new

        numPoints_tmp = len(xs)

        tmp2_xs = []
        tmp2_ys = []
        tmp2_zs = []
        tmp2_vs = []
        for i in range(len(xs)):
            # normalize by (x - mean_glb) / std_glb
            x = (xs[i] - x_mean_glb) / x_std_glb
            y = (ys[i] - y_mean_glb) / y_std_glb
            z = (zs[i] - z_mean_glb) / z_std_glb
            v = (vs[i] - v_mean_glb) / v_std_glb

            tmp2_xs.append(x)
            tmp2_ys.append(y)
            tmp2_zs.append(z)
            tmp2_vs.append(v)
        #endfor
        
        xc = [0] * g_cluster_num
        yc = [0] * g_cluster_num
        zc = [0] * g_cluster_num
        vc = [0] * g_cluster_num

        if normalizeFunc == 4:
            if cluster_func == 1:
                # Kmeans clustering
                c_labels, n_clusters = KmeanCluster(tmp2_xs, tmp2_ys, tmp2_zs, tmp2_vs)
            elif cluster_func == 2:
                # DBSCAN clustering
                c_labels, n_clusters = DbscanCluster(tmp2_xs, tmp2_ys, tmp2_zs, tmp2_vs)
            elif cluster_func == 3:
                # Geometry clustering
                if g_cluster_in_norm:
                    c_labels, n_clusters = GeometryCluster(tmp2_xs, tmp2_ys, tmp2_zs, tmp2_vs)
                else:
                    c_labels, n_clusters = GeometryCluster(xs, ys, zs, vs)
            else:
                c_labels = [0] * len(tmp2_xs)
                n_clusters = 1
            #endif

            if n_clusters > g_cluster_num:
                #print('frame: ', frameNum, 'cluster: ', n_clusters, '[', list(c_labels).count(0), list(c_labels).count(1), \
                #      list(c_labels).count(2), list(c_labels).count(3), list(c_labels).count(-1), ']')
                n_clusters = g_cluster_num
            #endif

            if n_clusters > 0:
                if g_cluster_num == 1:
                    n_clusters = 1

                    tmp_xs = []
                    tmp_ys = []
                    tmp_zs = []
                    tmp_vs = []

                    for i in range(len(c_labels)):
                        k = c_labels[i]  # cluster ID
                        if k >= 0:       # filter noise (-1)
                            tmp_xs.append(tmp2_xs[i])
                            tmp_ys.append(tmp2_ys[i])
                            tmp_zs.append(tmp2_zs[i])
                            tmp_vs.append(tmp2_vs[i])
                        #endif
                    #endfor

                    xc[0], yc[0], zc[0], vc[0] = filterFunc(tmp_xs, tmp_ys, tmp_zs, tmp_vs)
                else:
                    xc_tmp = [n_max_clip+1] * g_cluster_num
                    yc_tmp = [n_max_clip+1] * g_cluster_num
                    zc_tmp = [n_max_clip+1] * g_cluster_num
                    vc_tmp = [n_max_clip+1] * g_cluster_num

                    for j in range(n_clusters):
                        tmp_xs = []
                        tmp_ys = []
                        tmp_zs = []
                        tmp_vs = []

                        for i in range(len(c_labels)):
                            k = c_labels[i]  # cluster ID
                            if k == j:
                                tmp_xs.append(tmp2_xs[i])
                                tmp_ys.append(tmp2_ys[i])
                                tmp_zs.append(tmp2_zs[i])
                                tmp_vs.append(tmp2_vs[i])
                            #endif
                        #endfor

                        # cluster #j
                        xc_tmp[j], yc_tmp[j], zc_tmp[j], vc_tmp[j] = filterFunc(tmp_xs, tmp_ys, tmp_zs, tmp_vs)
                        #print(j, 'cluster: ', n_clusters, 'noise: ', list(c_labels).count(-1), xc)
                    #endfor

                    if g_sort_by_x == 1:
                        # sort by x-axis
                        c_centers = np.array([xc_tmp, yc_tmp, zc_tmp, vc_tmp]).T
                        c_centers = sorted(c_centers, key = lambda center : center[0])
                        c_centers = np.array(c_centers) # list to array conversion

                        for j in range(n_clusters):
                            if c_centers[j,0] < (n_max_clip+1):
                                xc[j] = c_centers[j,0]
                                yc[j] = c_centers[j,1]
                                zc[j] = c_centers[j,2]
                                vc[j] = c_centers[j,3]
                            #endif
                        #endfor
                    else:
                        for j in range(n_clusters):
                            if xc_tmp[j] < (n_max_clip+1):
                                xc[j] = xc_tmp[j]
                                yc[j] = yc_tmp[j]
                                zc[j] = zc_tmp[j]
                                vc[j] = vc_tmp[j]
                            #endif
                        #endfor
                    #endif
                #endif
            else:
                xc[0], yc[0], zc[0], vc[0] = filterFunc(tmp2_xs, tmp2_ys, tmp2_zs, tmp2_vs)
            #endif
        elif normalizeFunc == 5:
            tmp_xs = tmp2_xs
            tmp_ys = tmp2_ys
            tmp_zs = tmp2_zs
            tmp_vs = tmp2_vs        

            xc[0], yc[0], zc[0], vc[0] = filterFunc(tmp_xs, tmp_ys, tmp_zs, tmp_vs)
        else:
            # original point data
            tmp_xs = xs
            tmp_ys = ys
            tmp_zs = zs
            tmp_vs = vs

            xc[0], yc[0], zc[0], vc[0] = filterFunc(tmp_xs, tmp_ys, tmp_zs, tmp_vs)
        #endif

        # skip the non-action frame
        if frameNum > skip_frame:
            ff.append(frameNum)
            xx.append(xc)
            yy.append(yc)
            zz.append(zc)
            vv.append(vc)
        #endif
    else:
        print('read bin file NG', filename)
        retValue = 0
    #endif

    return retValue
#enddef genMotionWaveform()


def saveToCsv(xx:list, yy:list, zz:list, vv:list):
    global saveName

    # output features to CSV
    TIME_HEADER = 'Time(Seconds)'
    c_t = 0     # Current time
    dt = 1/hz
    tt = []
    for x in xx[:,0]:
        tt.append(round(c_t, 2))
        c_t += dt
    #endfor

    # create a dictionary with the X,Y,Z,V lists
    if g_cluster_num >= 8:
        # 8 group
        dict = {TIME_HEADER: tt,'X1':xx[:,0],'Y1':yy[:,0],'Z1':zz[:,0],'V1':vv[:,0],\
                                'X2':xx[:,1],'Y2':yy[:,1],'Z2':zz[:,1],'V2':vv[:,1],\
                                'X3':xx[:,2],'Y3':yy[:,2],'Z3':zz[:,2],'V3':vv[:,2],\
                                'X4':xx[:,3],'Y4':yy[:,3],'Z4':zz[:,3],'V4':vv[:,3],\
                                'X5':xx[:,4],'Y5':yy[:,4],'Z5':zz[:,4],'V5':vv[:,4],\
                                'X6':xx[:,5],'Y6':yy[:,5],'Z6':zz[:,5],'V6':vv[:,5],\
                                'X7':xx[:,6],'Y7':yy[:,6],'Z7':zz[:,6],'V7':vv[:,6],\
                                'X8':xx[:,7],'Y8':yy[:,7],'Z8':zz[:,7],'V8':vv[:,7]}    
    elif g_cluster_num >= 4:
        # 4 group
        dict = {TIME_HEADER: tt,'X1':xx[:,0],'Y1':yy[:,0],'Z1':zz[:,0],'V1':vv[:,0],\
                                'X2':xx[:,1],'Y2':yy[:,1],'Z2':zz[:,1],'V2':vv[:,1],\
                                'X3':xx[:,2],'Y3':yy[:,2],'Z3':zz[:,2],'V3':vv[:,2],\
                                'X4':xx[:,3],'Y4':yy[:,3],'Z4':zz[:,3],'V4':vv[:,3]}
    elif g_cluster_num == 3:
        # 3 group
        dict = {TIME_HEADER: tt,'X1':xx[:,0],'Y1':yy[:,0],'Z1':zz[:,0],'V1':vv[:,0],\
                                'X2':xx[:,1],'Y2':yy[:,1],'Z2':zz[:,1],'V2':vv[:,1],\
                                'X3':xx[:,2],'Y3':yy[:,2],'Z3':zz[:,2],'V3':vv[:,2]}
    elif g_cluster_num == 2:
        # 2 group
        dict = {TIME_HEADER: tt,'X1':xx[:,0],'Y1':yy[:,0],'Z1':zz[:,0],'V1':vv[:,0],\
                                'X2':xx[:,1],'Y2':yy[:,1],'Z2':zz[:,1],'V2':vv[:,1]}
    else:
        dict = {TIME_HEADER: tt, 'X': xx[:,0], 'Y': yy[:,0], 'Z': zz[:,0], 'V': vv[:,0]}
    #endif

    # create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(dict)

    # write the DataFrame to a CSV file
    df.to_csv(saveName + '.csv', index=False)
#enddef saveToCsv()


def drawMotionWaveform(xx:list, yy:list, zz:list, vv:list):
    global saveName
    global g_cluster_num

    # creating figures
    if g_cluster_num > 1:
        fig, ax = plt.subplots(g_cluster_num, 1, figsize=(30, 8.5))

        for i in range(g_cluster_num):
            # adding title and labels
            ax[i].set_title("Motion Waveform")
            ax[i].set_xlabel("frame")
            ax[i].set_ylabel("feature_" + str(i))

            # plot the feature
            ax[i].plot(ff, xx[:,i], c='mediumseagreen', label='x')
            ax[i].plot(ff, yy[:,i], c='cornflowerblue', label='y')
            ax[i].plot(ff, zz[:,i], c='orange', label='z')
            ax[i].plot(ff, vv[:,i], c='silver', label='v')
            ax[i].legend(fontsize=12)
        #endfor
    else:
        fig, ax = plt.subplots(1, 1, figsize=(30, 5))

        # adding title and labels
        ax.set_title("Motion Waveform")
        ax.set_xlabel("frame")
        ax.set_ylabel("feature")

        # plot the feature
        ax.plot(ff, xx[:,0], c='mediumseagreen', label='x')
        ax.plot(ff, yy[:,0], c='cornflowerblue', label='y')
        ax.plot(ff, zz[:,0], c='orange', label='z')
        ax.plot(ff, vv[:,0], c='silver', label='v')
        ax.legend(fontsize=12)
    #endif

    fig.tight_layout()

    # save the feature map
    plt.savefig(saveName + '.jpg')

    if g_all_track == 1:
        plt.show(block=False)
        plt.pause(3)
        plt.close('all')
    else:
        plt.show(block=True)
    #endif

#enddef drawMotionWaveform()


def singleTrack(i, j):
    global saveName
    global xs, ys, zs, vs, cs
    global ff, xx, yy, zz, vv
    global g_cluster_num
    global x_mean_glb, y_mean_glb, z_mean_glb, v_mean_glb
    global x_std_glb, y_std_glb, z_std_glb, v_std_glb
    global bin_idx
    global azi_idx, aziTilt

    # single track
    bin_idx = i
    azi_idx = j

    saveName = './temp/' + bin_path[bin_idx] + '__' + str(cluster_func) + '_' + str(g_cluster_num) + '_' + \
                str(persistent) + '_' + str(bKF) + '_' + str(normalizeFunc) + '_' + str(aziTilt[azi_idx])

    print('Config: ', bin_idx, azi_idx, saveName)

    xs = []
    ys = []
    zs = []
    vs = []
    cs = []
    ss = []

    ff = []
    xx = []
    yy = []
    zz = []
    vv = []

    last_numPoints = [0] * 20

    '''
    x_mean_glb = -0.14
    y_mean_glb = 1.65
    z_mean_glb = 1.2
    v_mean_glb = 0.3

    x_std_glb = 0.53
    y_std_glb = 0.5
    z_std_glb = 0.47
    v_std_glb = 0.27
    '''

    #'''
    x_mean_glb = 0.0
    y_mean_glb = 0.0
    z_mean_glb = 1.2
    v_mean_glb = 0.3

    x_std_glb = 1.42
    y_std_glb = 1.42
    z_std_glb = 0.47
    v_std_glb = 0.27
    #'''


    # normalize the feature
    ret = 1
    frameNum = 1
    while (ret == 1):
        ret = genMotionWaveform(frameNum)
        frameNum += 1
    #endwhile

    xx = np.array(xx) # list to array conversion
    yy = np.array(yy) # list to array conversion
    zz = np.array(zz) # list to array conversion
    vv = np.array(vv) # list to array conversion

    # smooth the featue waveform
    if len(xx) > 0:
        # iterate each cluster
        for i in range(g_cluster_num):
            # filter each channel
            xx[:,i] = KalmanF(xx[:,i]) # i_x cluster waveform
            yy[:,i] = KalmanF(yy[:,i]) # i_y cluster waveform
            zz[:,i] = KalmanF(zz[:,i])
            vv[:,i] = KalmanF(vv[:,i])
        #endfor
    #endif

    # output features to CSV
    saveToCsv(xx, yy, zz, vv)

    # draw MotionWaveform and save MotionWaveform to a PNG file
    drawMotionWaveform(xx, yy, zz, vv)

#enddef singleTrack()




# main
print('glb_mean: ', round(x_mean_glb,2), round(y_mean_glb,2), round(z_mean_glb,2), round(v_mean_glb,2))
print('glb_std:  ', round(x_std_glb,2), round(y_std_glb,2), round(z_std_glb,2), round(v_std_glb,2))

np.seterr(divide = 'ignore')

if g_all_track == 1:
    # all tracks
    for i in range(len(bin_path)):
        for j in range(len(aziTilt)):
            singleTrack(i, j)
        #endfor
    #endfor
else:
    for j in range(len(aziTilt)):
        singleTrack(bin_idx, j)
    #endfor
#endif

