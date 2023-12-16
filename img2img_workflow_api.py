import os
import json
import glob
import sys
from natsort import natsorted
from urllib import request, parse
import random

#  ==================================================================

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)    
    request.urlopen(req)   
#  ==================================================================

def get_images(folder_path):

    # check dir exists
    if not os.path.isdir(folder_path):
        print(f"Directory does not exist: {folder_path}")
        return []

    # Patterns for JPG and PNG 
    jpg_pattern = os.path.join(folder_path, '*.jpg')
    png_pattern = os.path.join(folder_path, '*.png')

    # Use glob to search for JPG and PNG files in the directory
    images_list = glob.glob(jpg_pattern, recursive=True) + glob.glob(png_pattern, recursive=True)

    images_list = natsorted(images_list)
    return images_list

#  ==================================================================


# load the workflow file, assign it to variable named prompt_workflow
prompt_workflow = json.load(open('workflow_api.json'))

# by default it expects your images to be in the 'ComfyUI/input' directory. 
# if it's anywhere else then you either need to put it in a path relative to 
# ComfyUI/input or use an absolute path if outside ComfyUI main directory. 

# image dir (linux)
image_dir = os.path.join('/home','johnl','Desktop','img2img_examples')
# for Windows use for example:
# image_dir = os.path.join('C:\\', 'Users', 'Y777', 'Desktop', 'img2img_examples')

# get list of images in the directory 
input_images = get_images(image_dir)

# Check if the list is empty and quit the script if it is
if not input_images:
    print("No images found. Exiting the script.")
    sys.exit()

# give some easy-to-remember names to the nodes
chkpoint_loader_node = prompt_workflow["14"]
prompt_pos_node = prompt_workflow["6"]
prompt_neg_node = prompt_workflow["7"]
load_image_node = prompt_workflow["10"]
ksampler_node = prompt_workflow["3"]
save_image_node = prompt_workflow["9"]

# load the checkpoint that we want. 
# make sure the path is correct to avoid 'HTTP Error 400: Bad Request' errors
chkpoint_loader_node["inputs"]["ckpt_name"] = "SD1-5/sd_v1-5_vae.ckpt"

save_image_node["inputs"]["filename_prefix"] = 'img2img_api'
prompt_pos_node["inputs"]["text"] = 'painting of a beautiful landscape, green mountains in the background, a river, lake, blue sky, clouds, in the style of Frank Bramley, artstation, concept art'
prompt_neg_node["inputs"]["text"] = ''
ksampler_node["inputs"]["steps"] = 25
ksampler_node["inputs"]["cfg"] = 10


for img in input_images:

    # set a random seed in KSampler node 
    ksampler_node["inputs"]["seed"] = random.randint(1, 1125899906842600)
    load_image_node["inputs"]["image"] = img
    ksampler_node["inputs"]["denoise"] = 0.7

    # everything set, add entire workflow to queue.
    queue_prompt(prompt_workflow)

    