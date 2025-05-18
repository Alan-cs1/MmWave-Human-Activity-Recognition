# 3D point cloud in Python using matplotlib

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

import sys

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

# Create point cloud data
xs = []
ys = []
zs = []
vs = []
cs = []


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
saveNum = 1

sensorHeight = 2.0
azi_idx = 0
elevTilt = 15
aziTilt = [0, 45, 90, 135, 180, 225, 270, 315, 360]
persistent = 5
last_numPoints = [0] * 20
g_cluster_num = 4
cluster_func = 3    # 2: DBSCAN, 3: Geometry 
g_c_centers = []

saveName = str(cluster_func) + '_' + str(persistent) + '_' + str(aziTilt[azi_idx]) +\
           '_' + bin_path[bin_idx]

video_rec = 0
g_all_track = 0

g_color_map = [[1, 0, 0, 1], [0, 0, 1, 1], [0.5, 0, 1, 1], [0, 1, 0, 1]]



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
        #print(c_labels)

        # cluster centers
        c_centers = Kmean.cluster_centers_
        #print(c_centers)

        # sort by x-axis
        c_centers = sorted(c_centers, key = lambda center : center[0])
        c_centers = np.array(c_centers) # list to array conversion
        #print(c_centers)

        return c_labels, c_centers
    else:
        return [], []
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

        db = DBSCAN(eps=0.3, min_samples=60).fit(pointcloud)
        c_labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
        n_noise = list(c_labels).count(-1)

        if n_clusters > g_cluster_num:
            print('cluster: ', n_clusters, 'noise: ', n_noise, 'n_label: ', len(c_labels))
            n_clusters = g_cluster_num
        #endif

        # calculate the cluster centers
        c_centers = []

        if n_clusters > 0:
            # calculate the mean of each cluster
            x_sum = [0] * n_clusters
            y_sum = [0] * n_clusters
            z_sum = [0] * n_clusters
            v_sum = [0] * n_clusters

            x_cnt = [0] * n_clusters
            y_cnt = [0] * n_clusters
            z_cnt = [0] * n_clusters
            v_cnt = [0] * n_clusters

            #print(c_labels)

            #unique_labels = set(c_labels)
            for i in range(len(c_labels)):
                k = c_labels[i]  # cluster ID

                if k >= 0 and k < g_cluster_num:
                    x_sum[k] += xs[i]
                    y_sum[k] += ys[i]
                    z_sum[k] += zs[i]
                    v_sum[k] += vs[i]

                    x_cnt[k] += 1
                    y_cnt[k] += 1
                    z_cnt[k] += 1
                    v_cnt[k] += 1
                #endif
            #endfor

            for k in range(n_clusters):
                x_mean = 0
                y_mean = 0
                z_mean = 0
                v_mean = 0

                if x_cnt[k] > 0:
                    x_mean = round(x_sum[k] / x_cnt[k],2)
                if y_cnt[k] > 0:
                    y_mean = round(y_sum[k] / y_cnt[k],2)
                if z_cnt[k] > 0:
                    z_mean = round(z_sum[k] / z_cnt[k],2)
                if v_cnt[k] > 0:
                    v_mean = round(v_sum[k] / v_cnt[k],2)

                c_centers.append([x_mean, y_mean, z_mean, v_mean])
            #endfor

            #print(c_centers)

            # sort by x-axis
            c_centers = sorted(c_centers, key = lambda center : center[0])
            c_centers = np.array(c_centers) # list to array conversion
            #print('cluster: ', n_clusters, c_centers)
        #endif

        return c_labels, c_centers
    else:
        return [], []
    #endif

#enddef DbscanCluster()


def GeometryCluster(xs:list, ys:list, zs:list, vs:list):
    global g_cluster_num

    c_labels = []
    n_clusters = g_cluster_num
    
    # calculate the center point
    xc = 0
    yc = 0
    zc = 0
    vc = 0
    if len(xs) > 0:
        xc = mean(xs)
        yc = mean(ys)
        zc = mean(zs)
        vc = mean(vs)
    #endif

    #print(xc, yc, zc, vc)

    x_threshold = xc
    y_threshold = yc
    z_threshold = zc  
    
    if len(zs) > 0:
        for i in range(len(zs)):
            if n_clusters == 2:           
                # 2 groups clustering
                if zs[i] >= z_threshold:
                    c_labels.append(0)
                else:
                    c_labels.append(1)   
                #endif
            elif n_clusters == 4:  
                # 4 groups clustering
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
                # 8 groups clustering
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
        
        #print(c_labels)
        
        # calculate the cluster centers
        c_centers = []

        if n_clusters > 0:
            # calculate the mean of each cluster
            x_sum = [0] * n_clusters
            y_sum = [0] * n_clusters
            z_sum = [0] * n_clusters
            v_sum = [0] * n_clusters

            x_cnt = [0] * n_clusters
            y_cnt = [0] * n_clusters
            z_cnt = [0] * n_clusters
            v_cnt = [0] * n_clusters

            #print(c_labels)

            #unique_labels = set(c_labels)
            for i in range(len(c_labels)):
                k = c_labels[i]  # cluster ID

                if k >= 0 and k < n_clusters:
                    x_sum[k] += xs[i]
                    y_sum[k] += ys[i]
                    z_sum[k] += zs[i]
                    v_sum[k] += vs[i]

                    x_cnt[k] += 1
                    y_cnt[k] += 1
                    z_cnt[k] += 1
                    v_cnt[k] += 1
                #endif
            #endfor

            for k in range(n_clusters):
                x_mean = 0
                y_mean = 0
                z_mean = 0
                v_mean = 0

                if x_cnt[k] > 0:
                    x_mean = round(x_sum[k] / x_cnt[k],2)
                if y_cnt[k] > 0:
                    y_mean = round(y_sum[k] / y_cnt[k],2)
                if z_cnt[k] > 0:
                    z_mean = round(z_sum[k] / z_cnt[k],2)
                if v_cnt[k] > 0:
                    v_mean = round(v_sum[k] / v_cnt[k],2)

                c_centers.append([x_mean, y_mean, z_mean, v_mean])
            #endfor

            #print(c_centers)

            # sort by x-axis
            #c_centers = sorted(c_centers, key = lambda center : center[0])
            c_centers = np.array(c_centers) # list to array conversion
            #print('cluster: ', n_clusters, c_centers)
        #endif

        return c_labels, c_centers
    else:
        return [], []
    #endif   
    
#enddef GeometryCluster()


class draw_pointcloud():
    def __init__(self):
        global bin_idx

        if g_all_track == 1:
            bin_idx = 0
        #endif        

        while True:
            self.singleTrack(bin_idx)

            if g_all_track == 1:
                if bin_idx >= len(bin_path):
                #if bin_idx >= 1:
                    sys.exit(0)
                #endif
                
                bin_idx += 1
            else:
                sys.exit(0)
            #endif
        #endwhile

    def readPointCloud(self):
        global xs, ys, zs, vs, cs
        global saveNum, skip_frame
        global numPoints
        global last_numPoints
        global g_c_centers
        global bin_idx, g_all_track


        while True:
            filename = './binData/' + bin_path[bin_idx] + '/pHistBytes_' + str(saveNum) + '.bin'

            try:
                dfile = open(filename, 'rb', 0)
                #print('open ', filename)
            except:
                print('cannot open ', filename)
                
                self.running = False           
                yield []

                return 0
            #endtry

            frameData = bytes(list(dfile.read()))
            if (frameData):
                outputDict = parseStandardFrame(frameData)

                frameNum = outputDict['frameNum']

                # Number of Points
                if ('numDetectedPoints' in outputDict):
                    numPoints = outputDict['numDetectedPoints']
                else:
                    numPoints = 0
                #endif

                # remove the oldest frame from persistent window
                xs[:last_numPoints[saveNum%persistent]] = []
                ys[:last_numPoints[saveNum%persistent]] = []
                zs[:last_numPoints[saveNum%persistent]] = []
                vs[:last_numPoints[saveNum%persistent]] = []
                cs[:last_numPoints[saveNum%persistent]] = []

                numPoints_new = 0

                if (numPoints):
                    # Point Cloud
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

                                #doppler = np.clip(abs(pointCloud[i,3]),0,1)
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

                                #cs.append([doppler, 1, 0, 1])
                                #cs.append([0, snr, 0, 1])
                                #cs.append([doppler, snr, 0, 1])
                                if abs(doppler) >= 0.1:
                                    cs.append([1, 0, 0, 1])
                                else:
                                    cs.append([0, 1, 0, 1])
                                #endif

                                numPoints_new += 1
                            #endfor

                            #print('frame ', outputDict['frameNum'], '[', numPoints, ']', filename)
                            #print('frame ', outputDict['frameNum'], '[', len(xs),']', filename)
                        #endif
                    #endif
                #else:
                #    print('frame ', outputDict['frameNum'], '[', numPoints, 0, ']')
                #endif
            else:
                frameNum = -1
                print('read bin file NG')
            #endif

            last_numPoints[frameNum % persistent] = numPoints_new

            # skip zero point cloud
            if len(xs) > 0:
                # skip the non-action frame
                if saveNum > skip_frame:
                    if cluster_func == 1:
                        # Kmeans clustering
                        c_labels, c_centers = KmeanCluster(xs, ys, zs, vs)
                    elif cluster_func == 2:
                        # DBSCAN clustering
                        c_labels, c_centers = DbscanCluster(xs, ys, zs, vs)
                    elif cluster_func == 3:
                        # Geometry clustering
                        c_labels, c_centers = GeometryCluster(xs, ys, zs, vs)                          
                    else:
                        # no clustering
                        c_labels = []
                        c_centers = []

                        if len(xs) > 0:
                            x_mean = mean(xs)
                            y_mean = mean(ys)
                            z_mean = mean(zs)
                            v_mean = mean(vs)

                            c_labels = [0] * len(xs)
                            c_centers = [[x_mean, y_mean, z_mean, v_mean]]
                            c_centers = np.array(c_centers) # list to array conversion
                        #endif
                    #endif

                    g_c_centers = c_centers
                    yield c_labels
                #endif
            #endif

            saveNum += 1
        #endwhile

        return 1
    #enddef readPointCloud()


    def update(self, c_labels):
        global xs, ys, zs, vs, cs
        global saveNum
        global g_c_centers
        global g_color_map


        if self.running == False:
            plt.close(self.fig)
            return 0
        #endif

        xs_tmp = []
        ys_tmp = []
        zs_tmp = []
        cs_tmp = []

        if len(c_labels) > 0:
            for i in range(len(c_labels)):
                k = c_labels[i]  # cluster ID

                if k >= 0 and k < g_cluster_num:
                #if k >= 0:
                    xs_tmp.append(xs[i])
                    ys_tmp.append(ys[i])
                    zs_tmp.append(zs[i])
                    cs_tmp.append(g_color_map[k % 4])
                #endif
            #endfor  
        else:
            return 1
        #endif

        self.ax.clear()

        '''
        c_centers = g_c_centers
        if len(c_centers) > 0:
            img = self.ax.scatter(c_centers[:, 0], c_centers[:, 1], c_centers[:, 2], c='gray', s=100, alpha=0.5);
        #endif
        '''

        # point cloud
        img = self.ax.scatter(xs_tmp, ys_tmp, zs_tmp, marker='s', s=5, c=cs_tmp, alpha=0.5)     # c = doppler

        # format plot
        # adding title and labels
        self.ax.set_title("3D Point Cloud" + ' ' + str(saveNum) + ' [' + \
                      str(list(c_labels).count(0)) + ',' + str(list(c_labels).count(1)) + ',' +\
                      str(list(c_labels).count(2)) + ',' + str(list(c_labels).count(3)) + ',' +\
                      str(list(c_labels).count(4)) + ',' + str(list(c_labels).count(5)) + ',' +\
                      str(list(c_labels).count(6)) + ',' + str(list(c_labels).count(7)) + ',' +\
                      str(list(c_labels).count(-1)) + ']', fontsize=40)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_xlim([-3.5,3.5])
        self.ax.set_ylim([0,5])
        self.ax.set_zlim([0,3.0])
        self.ax.set_xticks([-3.0,-1.5,0,1.5,3.0])
        self.ax.set_yticks([1,3,5])
        self.ax.set_zticks([0.5,1,1.5,2,2.5])

        return 1
    #enddef update()


    def singleTrack(self, i):
        global saveNum, saveName
        global video_rec

        if i >= len(bin_path):
            sys.exit(0)
            return
        #endif

        saveName = str(cluster_func) + '_' + str(persistent) + '_' + str(aziTilt[azi_idx]) +\
                   '_' + bin_path[i]

        saveNum = 1

        xs = []
        ys = []
        zs = []
        vs = []
        cs = []

        last_numPoints = [0] * 20
        
        self.running = True
        
        print(f"{i:2d}, {saveName}")
        
        # creating figures
        self.fig = plt.figure(figsize=(25, 25))
        self.ax = self.fig.add_subplot(111, projection='3d')

        #XZ
        self.ax.view_init(elev=0, azim=-90)
        #YZ
        #self.ax.view_init(elev=0, azim=0)
        #XY
        #self.ax.view_init(elev=90, azim=-90)

        # adding title and labels
        self.ax.set_title("3D Point Cloud", fontsize=40)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_xlim([-3.5,3.5])
        self.ax.set_ylim([0,5])
        self.ax.set_zlim([0,3.0])
        self.ax.set_xticks([-3.0,-1.5,0,1.5,3.0])
        self.ax.set_yticks([1,3,5])
        self.ax.set_zticks([0.5,1,1.5,2,2.5])
        
        # Construct the animation, using the update function as the animation director.
        self.ani = animation.FuncAnimation(self.fig, self.update, self.readPointCloud(), \
                   fargs=(), interval=50, save_count=10000, repeat = False)

        if video_rec == 1:
            FFwriter = animation.FFMpegWriter(fps=18, extra_args=['-vcodec', 'libx264'])
            self.ani.save('XY' + '_' + saveName + '.mp4', writer = FFwriter)
        #endif

        # displaying plot
        plt.show()
    #end_singleTrack()



#==================================================================================================
#
#==================================================================================================
# main


plot_3dpointcloud = draw_pointcloud()




