import sys
import time
import vrep # access all the VREP elements


# connect to server
vrep.simxFinish(-1) # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # start a connection
if clientID != -1:
    print("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")


# get motors handles
# err_code,l_motor_handle = vrep.simxGetObjectHandle(clientID,"bubbleRob_leftMotor", vrep.simx_opmode_blocking)
# err_code,r_motor_handle = vrep.simxGetObjectHandle(clientID,"bubbleRob_rightMotor", vrep.simx_opmode_blocking)
# print(l_motor_handle)
# print(r_motor_handle)

# applying velocities to joints motors
# err_code = vrep.simxSetJointTargetVelocity(clientID,l_motor_handle,1.0, vrep.simx_opmode_streaming)
# err_code = vrep.simxSetJointTargetVelocity(clientID,r_motor_handle,1.0, vrep.simx_opmode_streaming)

# proximity sensing
# err_code,ps_handle = vrep.simxGetObjectHandle(clientID,"bubbleRob_sensingNose", vrep.simx_opmode_blocking)
# err_code,detectionState,detectedPoint,detectedObjectHandle, \
#     detectedSurfaceNormalVector=vrep.simxReadProximitySensor( \
#             clientID, ps_handle,vrep.simx_opmode_streaming )

# print(detectedPoint)
# print(detectedObjectHandle)
# print(detectedSurfaceNormalVector)

import numpy as np #do this at the top of the program.
# detectedPointDist = np.linalg.norm(detectedPoint)
# print(detectedPointDist)

t = time.time() #record the initial time
# while(time.time()-t)<10: #run for 20 seconds
#     sensor_val = np.linalg.norm(detectedPoint)
#     if sensor_val < 0.2 and sensor_val>0.01:
#         l_steer = -1/sensor_val
#     else:
#         l_steer = 1.0
#
#     # apply velocities
#     err_code = vrep.simxSetJointTargetVelocity(clientID,l_motor_handle, l_steer,vrep.simx_opmode_streaming)
#     err_code = vrep.simxSetJointTargetVelocity(clientID,r_motor_handle, 1.0,vrep.simx_opmode_streaming)
#     time.sleep(0.2)
#
#     # sense
#     err_code,detectionState,detectedPoint,detectedObjectHandle, detectedSurfaceNormalVector = \
#                     vrep.simxReadProximitySensor(clientID, ps_handle,vrep.simx_opmode_buffer)
#     print (sensor_val,detectedPoint)


## vision sensing

err_code,camera = vrep.simxGetObjectHandle(clientID,"Vision_sensor2",vrep.simx_opmode_oneshot_wait)
print("1 error", err_code)
err_code,imagePlane = vrep.simxGetObjectHandle(clientID,"Image_plane",vrep.simx_opmode_blocking)
print("2 error", err_code)
err_code,resolution,image = vrep.simxGetVisionSensorImage(clientID, camera,1,vrep.simx_opmode_streaming)
print("3 error", err_code)
print(resolution)
print(len(image))
t = time.time() #record the initial time
import matplotlib.pyplot as mlp
while(time.time()-t)<10: #run for 20 seconds
  err_code,resolution,image = vrep.simxGetVisionSensorImage(clientID, camera,1,vrep.simx_opmode_buffer)
  # time.sleep(0.5)
  print("6 error", err_code)
  print(resolution)
  print(len(image))
  if image:
    img = np.array(image, dtype = np.uint8)
    img.resize([resolution[0],resolution[1]])
    mlp.pause(0.0001)
    mlp.imshow(img,origin="lower")

# err_code = vrep.simxSetObjectPosition(clientID, imagePlane, vrep.sim_handle_parent,[0,0,0], vrep.simx_opmode_oneshot)
# print("3 error", err_code)

# error = vrep.simxSetObjectOrientation(clientID, imagePlane, vrep.sim_handle_parent,[0,0.785398,0], vrep.simx_opmode_oneshot)
# print("4 error", err_code)
#
# err_code,resolution,image = vrep.simxGetVisionSensorImage(clientID, camera,1000,vrep.simx_opmode_streaming)
# print("5 error", err_code)


# get vision sensor image and render it

# print(resolution)


vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
print("Done")

