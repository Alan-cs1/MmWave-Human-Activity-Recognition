import struct
import sys
import serial
import binascii
import time
import numpy as np
import math

import os
import datetime

# Local File Imports
from parseTLVs import *
from gui_common import *

def parseStandardFrame(frameData):
    headerStruct = 'Q8I'
    frameHeaderLen = struct.calcsize(headerStruct)
    tlvHeaderLength = 8

    outputDict = {}
    outputDict['error'] = 0

    try:
        # Read in frame Header
        magic, version, totalPacketLen, platform, frameNum, timeCPUCycles, numDetectedObj, numTLVs, subFrameNum = struct.unpack(headerStruct, frameData[:frameHeaderLen])
    except:
        print('Error: Could not read frame header')
        outputDict['error'] = 1

    # Move frameData ptr to start of 1st TLV    
    frameData = frameData[frameHeaderLen:]

    # Save frame number to output
    outputDict['frameNum'] = frameNum

    # print("")
    # print ("FrameNum: ", frameNum)

    # Initialize the point cloud struct since it is modified by multiple TLV's
    # Each point has the following: X, Y, Z, Doppler, SNR, Noise, Track index
    outputDict['pointCloud'] = np.zeros((numDetectedObj, 7), np.float64)
    # Initialize the track indexes to a value which indicates no track
    outputDict['pointCloud'][:, 6] = 255
    # Find and parse all TLV's
    for i in range(numTLVs):
        # print ("Frame Data at start of TLV: ", frameData[:10])
        try:
            tlvType, tlvLength = tlvHeaderDecode(frameData[:tlvHeaderLength])
            frameData = frameData[tlvHeaderLength:]
            # tlvLength = tlvLength - tlvHeaderLength
        except:
            print('TLV Header Parsing Failure')
            outputDict['error'] = 2
        # print ("TLV Number: ", tlvType)
        # print ("Frame Data before tlv parse: ", frameData[:10])

        if (outputDict['error'] == 2):
            print ("Ignored frame due to parsing error")
            break
        # Detected Points
        if (tlvType == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parsePointCloudTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        # Range Profile
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RANGE_PROFILE):
            outputDict['rangeProfile'] = parseRangeProfileTLV(frameData[:tlvLength])
        # Range Profile
        elif (tlvType == MMWDEMO_OUTPUT_EXT_MSG_RANGE_PROFILE_MAJOR):
            outputDict['rangeProfileMajor'] = parseRangeProfileTLV(frameData[:tlvLength])
        # Range Profile
        elif (tlvType == MMWDEMO_OUTPUT_EXT_MSG_RANGE_PROFILE_MINOR):
            outputDict['rangeProfileMinor'] = parseRangeProfileTLV(frameData[:tlvLength])
        # Noise Profile
        elif (tlvType == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE):
            pass
        # Static Azimuth Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP):
            pass
        # Range Doppler Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP):
            pass
        # Performance Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_STATS):
            pass
        # Side Info
        elif (tlvType == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO):
            outputDict['pointCloud'] = parseSideInfoTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
         # Azimuth Elevation Static Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP):
            pass
        # Temperature Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS):
            pass
        # Spherical Points
        elif (tlvType == MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parseSphericalPointCloudTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        # Target 3D
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST):
            outputDict['numDetectedTracks'], outputDict['trackData'] = parseTrackTLV(frameData[:tlvLength], tlvLength)
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT):
            outputDict['numDetectedHeights'], outputDict['heightData'] = parseTrackHeightTLV(frameData[:tlvLength], tlvLength)
         # Target index
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX):
            outputDict['trackIndexes'] = parseTargetIndexTLV(frameData[:tlvLength], tlvLength)
         # Capon Compressed Spherical Coordinates
        elif (tlvType == MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parseCompressedSphericalPointCloudTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        # Presence Indication
        elif (tlvType == MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION):
            pass
        # Occupancy State Machine
        elif (tlvType == MMWDEMO_OUTPUT_MSG_OCCUPANCY_STATE_MACHINE):
            outputDict['occupancy'] = parseOccStateMachTLV(frameData[:tlvLength])
        elif (tlvType == MMWDEMO_OUTPUT_MSG_VITALSIGNS):
            outputDict['vitals'] = parseVitalSignsTLV(frameData[:tlvLength], tlvLength)
        elif(tlvType == MMWDEMO_OUTPUT_EXT_MSG_DETECTED_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parsePointCloudExtTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        # Performance Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_EXT_STATS):
            pass
            
        
        else:
            print ("Warning: invalid TLV type: %d" % (tlvType))

        # print ("Frame Data after tlv parse: ", frameData[:10])
        # Move to next TLV
        frameData = frameData[tlvLength:]
        # print ("Frame Data at end of TLV: ", frameData[:10])
    return outputDict


# Capon Processing Chain uses a modified header with a slightly different set of TLV's, so it needs its own frame parser
# def parseCaponFrame(frameData):
#     tlvHeaderLength = 8
#     headerLength = 48
#     headerStruct = 'Q9I2H'
    
#     outputDict = {}
#     outputDict['error'] = 0

#     try:
#         magic, version, packetLength, platform, frameNum, subFrameNum, chirpMargin, frameMargin, uartSentTime, trackProcessTime, numTLVs, checksum =  struct.unpack(headerStruct, frameData[:headerLength])
#     except Exception as e:
#         print('Error: Could not read frame header')
#         outputDict['error'] = 1

#     outputDict['frameNum'] = frameNum        
#     frameData = frameData[headerLength:]
#     # Check TLVs
#     for i in range(numTLVs):
#         #try:
#         #print("DataIn Type", type(dataIn))
#         try:
#             tlvType, tlvLength = tlvHeaderDecode(frameData[:tlvHeaderLength])
#             frameData = frameData[tlvHeaderLength:]
#             dataLength = tlvLength - tlvHeaderLength
#         except:
#             print('TLV Header Parsing Failure')
#             outputDict['error'] = 2
        
#         # OOB Point Cloud
#         if (tlvType == 1): 
#             pass
#         # Range Profile
#         elif (tlvType == 2):
#             pass
#         # Noise Profile
#         elif (tlvType == 3):
#             pass
#         # Static Azimuth Heatmap
#         elif (tlvType == 4):
#             pass
#         # Range Doppler Heatmap
#         elif (tlvType == 5):
#             pass
#         # Capon Polar Coordinates
#         elif (tlvType == 6):
#             numDetectedPoints, parsedPointCloud = parseCaponPointCloudTLV(frameData[:dataLength], dataLength)
#             outputDict['pointCloudCapon'] = parsedPointCloud
#             outputDict['numDetectedPoints'] = numDetectedPoints
#         # Target 3D
#         elif (tlvType == 7):
#             numDetectedTracks, parsedTrackData = parseTrackTLV(frameData[:dataLength], dataLength)
#             outputDict['trackData'] = parsedTrackData
#             outputDict['numDetectedTracks'] = numDetectedTracks
#          # Target index
#         elif (tlvType == 8):
#             #self.parseTargetAssociations(dataIn[:dataLength])
#             outputDict['trackIndexCapon'] = parseTargetIndexTLV(frameData[:dataLength], dataLength)
#         # Classifier Output
#         elif (tlvType == 9):
#             pass
#         # Stats Info
#         elif (tlvType == 10):
#             pass
#         # Presence Indicator
#         elif (tlvType == 11):
#             pass
#         else:
#             print ("Warning: invalid TLV type: %d" % (tlvType))
        
#         frameData = frameData[dataLength:]
#     return outputDict



# Decode TLV Header
def tlvHeaderDecode(data):
    tlvType, tlvLength = struct.unpack('2I', data)
    return tlvType, tlvLength

