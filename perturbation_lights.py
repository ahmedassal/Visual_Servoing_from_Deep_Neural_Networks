import numpy as np
import vrep
import cv2


def setLightRandParams(clientID, light, max_light_intensity, max_light_distance, max_sigma):
  amplitude = np.random.random() * max_light_intensity
  light_dist_rand = np.random.random() * max_light_distance
  sigma = [np.random.random() * max_sigma, np.random.random() * max_sigma]
  pos_x = np.random.random() * 2 * light_dist_rand/np.sqrt(3) - light_dist_rand/np.sqrt(3)
  pos_y = -np.random.random() * light_dist_rand / np.sqrt(3)
  pos_z = -np.sqrt(light_dist_rand**2 - pos_x**2 + pos_y**2)
  length = np.sqrt(pos_x ** 2 + pos_y ** 2 + pos_z ** 2)
  pos = [pos_x, pos_y, pos_z]
  proj = [pos_x, pos_y]
  errSetPos = vrep.simxSetObjectPosition(clientID, light['handle'], vrep.sim_handle_parent , pos, vrep.simx_opmode_oneshot_wait)
  errGetPos, newPos = vrep.simxGetObjectPosition(clientID, light['handle'], vrep.sim_handle_parent , vrep.simx_opmode_oneshot_wait)
  length = np.sqrt(newPos[0] ** 2 + newPos[1] ** 2 + newPos[2] ** 2)
  if errSetPos :
    print("Failed to set position for light: {}, got error code: {}".format(light['handle'], errSetPos))
  elif errGetPos:
    print("Failed to get position for light: {}, got error code: {}".format(light['handle'], errGetPos))
  else:
    return amplitude, length, newPos, proj, sigma
  # print( length, newPos)
  return amplitude, length, newPos, proj, sigma

def buildLights(clientID, lights_names, light_handles, max_light_intensity, max_light_distance, max_sigma):
  lights ={}
  for light_name in lights_names:
    light={}
    light['name'] = light_name
    light['handle'] = light_handles[light_name]

    light['amplitude'], light['distance'], light['pos'], light['proj'], light['sigma']= \
        setLightRandParams(clientID, light, max_light_intensity, max_light_distance, max_sigma)
    lights[light_name] = light
  return lights

def projectLights(clientID, lights):
  for light_key, light_params in lights.items():
    light_params['proj_x'] = light_params['pos'][0]
    light_params['proj_y'] = light_params['pos'][1]
  # for light_key, light_params in lights.items():
  #   print(light_key)
  #   print(light_params)
  return lights

def f_l(image, amplitude, light_proj, sigma, bbox_ext, res):
  # print(amplitude, light_proj, sigma, bbox_ext, res)
  f = np.zeros(np.shape(image))
  inc_x = bbox_ext[0] / res[0]
  inc_y = bbox_ext[1] / res[1]

  for x in range(np.shape(image)[0]):
    for y in range(np.shape(image)[1]):
      x_ = -(x - res[0] / 2) * inc_x
      y_ = -(y - res[1] / 2) * inc_y

      s0 = 2*sigma[0]**2
      s1 = 2*sigma[1]**2
      f[x,y] = amplitude * np.exp(-(((x_-light_proj[0])/s0 ) + ((y_-light_proj[1])/s1)))
  # cv2.imshow("f_l", f)
  # print("Mean f", np.mean(f))
  return f

def computeLightsContributions(lights, image, bbox_ext, res):
  # print("Mean image", np.mean(image))
  image_l = np.zeros(np.shape(image))
  # print(np.shape(image))
  for light_key, light in lights.items():
    # image_l = np.add(image_l, f_l(image, light['amplitude'], light['proj'], light['sigma'], bbox_ext, res))
    image_l = np.add(image_l, np.multiply(image, f_l(image, light['amplitude'], light['proj'], light['sigma'], bbox_ext, res)))
    # cv2.imshow("image_l", image_l)
    # print("Mean image__l", np.mean(image_l))
  out= np.multiply(image, image_l)
  # out = cv2.normalize(out.astype('float'), None, 0, 255, cv2.NORM_MINMAX)
  out = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
  # print("out max", np.max(out[:]))
  # print("out min", np.min(out[:]))
  # print("out type", type(out))
  # print("out mean", np.mean(out))

  return out

def im2double(im):
  info = np.iinfo(im.dtype)  # Get the data type of the input image
  return im.astype(np.float) / info.max  # Divide all values by the largest possible value in the datatype

# val = setLightRandPose(10)
# dist=np.sqrt(val[1][0]**2 + val[1][1]**2 + val[1][2]**2)
#
# print(val)
# print(val[0])
# print(dist)
# print(val[0]==dist)