import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

import pickle
import json

# STEP 1 : Import image
step_1_time = time.time()
PROJECT_DIR = '/root/Dataset/smplify-x/'

RES_DIR = os.path.join(PROJECT_DIR, 'data')
FRAMES_DIR = os.path.join(RES_DIR, 'images')

def load_img(img_path):

  return np.asarray(Image.open(img_path))/255

target_name = "female-5"

test_img_path = FRAMES_DIR
filename = target_name + '.jpg'

test_img = load_img(os.path.join(test_img_path, filename))
step_1_time = time.time() - step_1_time
# print('STEP 1 : Import image, {}'.format(step_1_time))

# Visualization
# plt.figure(figsize=(5, 10))
# plt.title("Sample image visualization")
# plt.imshow(test_img)



# STEP 2 : Run Openpose

# @title Run OpenPose on the extracted frames
step_2_time = time.time()
os.system('cd /root/Dataset/smplify-x/')
KEYPOINTS_DIR = os.path.join(PROJECT_DIR, 'data', 'keypoints')
OPENPOSE_IMAGES_DIR = os.path.join(PROJECT_DIR , 'data' , 'openpose_images')

os.system('mkdir {}'.format(KEYPOINTS_DIR))
os.system('mkdir {}'.format(OPENPOSE_IMAGES_DIR))

print(FRAMES_DIR, KEYPOINTS_DIR)
os.system('cd openpose && ./build/examples/openpose/openpose.bin --image_dir {} --write_json {} --face --hand --display 0   --write_images {}'.format(FRAMES_DIR, KEYPOINTS_DIR, OPENPOSE_IMAGES_DIR))

input_img_path = os.path.join(FRAMES_DIR, sorted(os.listdir(FRAMES_DIR))[0])
print(sorted(os.listdir(OPENPOSE_IMAGES_DIR)))
openpose_img_path = os.path.join(OPENPOSE_IMAGES_DIR, sorted(os.listdir(OPENPOSE_IMAGES_DIR))[0])

test_img = load_img(input_img_path)
open_pose_img = load_img(openpose_img_path)

# Visualization
# plt.figure(figsize=(10, 10))
# plt.title("Input Frame + Openpose Prediction")
# plt.imshow(np.concatenate([test_img, open_pose_img], 1))

step_2_time = time.time() - step_2_time
# print('STEP 2 : Run Openpose, {}'.format(step_2_time))




# STEP 3 : Run SMPLify-X
step_3_time = time.time()
MODEL_PATH = '/root/Dataset/smplify-x/models/'
SMPLX_ZIP_PATH = MODEL_PATH + 'models_smplx_v1_1.zip' # @param {type:"string"}
VPOSER_ZIP_PATH = MODEL_PATH + 'vposer_v1_0.zip' # @param {type:"string"}

SMPLX_MODEL_PATH = '/root/Dataset/smplify-x/smplx'
VPOSER_MODEL_PATH = '/root/Dataset/smplify-x/vposer'

gender = 'female' #@param ["neutral", "female", "male"]

os.system('rm -rf /root/Dataset/smplify-x/data/smplifyx_results')
os.system('cd /root/Dataset/smplify-x/')

os.system('python smplifyx/main.py --config cfg_files/fit_smplx.yaml \
    --data_folder  /root/Dataset/smplify-x/data \
    --output_folder /root/Dataset/smplify-x/data/smplifyx_results \
    --visualize=False \
    --gender={} \
    --model_folder /root/Dataset/smplify-x/smplx/models \
    --vposer_ckpt /root/Dataset/smplify-x/vposer/vposer_v1_0 \
    --part_segm_fn smplx_parts_segm.pkl \
    --interpenetration=False'.format(gender))

step_3_time = time.time() - step_3_time
# print('STEP 3 : Run SMPLify-X, {}'.format(step_3_time))



# STEP 4 : Save to .json file
step_4_time = time.time()

with open('/root/Dataset/smplify-x/data/smplifyx_results/results/'+target_name+'/000.pkl', 'rb') as f:
    data = pickle.load(f)

for key in data:
    data[key] = data[key].tolist()

with open('./results/model_parameters_' + target_name + '.json','w') as f:
  json.dump(data, f, ensure_ascii=False, indent=4)

step_4_time = time.time() - step_4_time
# print('STEP 4 : Save to .json file {}'.format(step_4_time))


print('STEP 1 : Import image, {:.3f}s'.format(step_1_time))
print('STEP 2 : Run Openpose, {:.3f}s'.format(step_2_time))
print('STEP 3 : Run SMPLify-X, {:.3f}s'.format(step_3_time))
print('STEP 4 : Save to .json file {:.3f}s'.format(step_4_time))
print('-'*20)
print('Total processing time {:.3f}s'.format(step_1_time + step_2_time + step_3_time + step_4_time))