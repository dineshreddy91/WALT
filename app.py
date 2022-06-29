import numpy as np
import torch
import gradio as gr
from infer import detections

import os
os.system("mkdir data")
os.system("mkdir data/models")
if not os.path.exists("data/models/walt_people.pth"):
    os.system("wget https://www.cs.cmu.edu/~walt/models/walt_people.pth -O data/models/walt_people.pth")
if not os.path.exists("data/models/walt_vehicle.pth"):
    os.system("wget https://www.cs.cmu.edu/~walt/models/walt_vehicle.pth -O data/models/walt_vehicle.pth")
'''
'''
def walt_demo(input_img, confidence_threshold):
    #detect_people = detections('configs/walt/walt_people.py', 'cuda:0', model_path='data/models/walt_people.pth')
    if torch.cuda.is_available() == False:
        device='cpu'
    else:
        device='cuda:0'
    #detect_people = detections('configs/walt/walt_people.py', device, model_path='data/models/walt_people.pth')
    detect = detections('configs/walt/walt_vehicle.py', device, model_path='data/models/walt_vehicle.pth', threshold=confidence_threshold)
 
    count = 0
    #img = detect_people.run_on_image(input_img)
    output_img = detect.run_on_image(input_img)
    #try:
    #except:
    #    print("detecting on image failed")

    return output_img

description = """
WALT Demo on WALT dataset. After watching and automatically learning for several days, this approach shows significant performance improvement in detecting and segmenting occluded people and vehicles, over human-supervised amodal approaches</b>.
<center>
    <a href="https://www.cs.cmu.edu/~walt/">
        <img style="display:inline" alt="Project page" src="https://img.shields.io/badge/Project%20Page-WALT-green">
    </a>
    <a href="https://www.cs.cmu.edu/~walt/pdf/walt.pdf"><img style="display:inline" src="https://img.shields.io/badge/Paper-Pdf-red"></a>
    <a href="https://github.com/dineshreddy91/WALT"><img style="display:inline" src="https://img.shields.io/github/stars/dineshreddy91/WALT?style=social"></a>
</center>
"""
title = "WALT:Watch And Learn 2D Amodal Representation using Time-lapse Imagery"
article="""
<center>
    <img src='https://visitor-badge.glitch.me/badge?page_id=anhquancao.MonoScene&left_color=darkmagenta&right_color=purple' alt='visitor badge'>
</center>
"""

examples = [
    ['demo/images/img_1.jpg',0.8],
    ['demo/images/img_2.jpg',0.8],
    ['demo/images/img_4.png',0.85],
]

'''
import cv2
filename='demo/images/img_1.jpg'
img=cv2.imread(filename)
img=walt_demo(img)
cv2.imwrite(filename.replace('/images/','/results/'),img)
cv2.imwrite('check.png',img)
'''
confidence_threshold = gr.Slider(minimum=0.3,
                                    maximum=1.0,
                                    step=0.01,
                                    value=1.0,
                                    label="Amodal Detection Confidence Threshold")
inputs = [gr.Image(), confidence_threshold]
demo = gr.Interface(walt_demo, 
        outputs="image",
        inputs=inputs, 
        article=article,
        title=title,
        enable_queue=True,
        examples=examples,
        description=description)

#demo.launch(server_name="0.0.0.0", server_port=7000)
demo.launch(share=True)


