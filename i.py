import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, net, utils

from skimage import io
import dlib

def parse_args(): #arguments to be parsed
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', 
        dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--snapshot', 
        dest='snapshot', help='Path of model snapshot.', default='mod29.pkl', type=str)
    parser.add_argument('--face_model', 
        dest='face_model', help='Path of DLIB face detection model.', 
        default='', type=str)
    parser.add_argument('--video', 
        dest='video_path', help='Path of video')
    parser.add_argument('--output_string', 
        dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', 
        dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', 
        dest='fps', help='Frames per second of source video', 
        type=float, default=30.)
    parser.add_argument('--arch', 
        dest='arch', 
        help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], '
            'ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True #enable cudnn 

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_path = args.video_path

    if not os.path.exists(out_dir): #make folder if not exist
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path): #check if video file exist
        sys.exit('Video does not exist')

    # Base network structure
    if args.arch == 'ResNet18':
        model = net.net(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
    elif args.arch == 'ResNet34':
        model = net.net(
            torchvision.models.resnet.BasicBlock, [3,4,6,3], 66)
    elif args.arch == 'ResNet101':
        model = net.net(
            torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
    elif args.arch == 'ResNet152':
        model = net.net(
            torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], 66)
    
    else:
        if args.arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = net.net(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1("face3.dat")

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) #crop center part of image and scale image 

    model.cuda(gpu) #use gpu

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    video = cv2.VideoCapture(args.video_path) #read video

    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, args.fps, (width, height))

    

    txt_out = open('output/video/output-%s.txt' % args.output_string, 'w')

    frame_num = 1
#video.isOpened()
    while (video.isOpened()):
        print(frame_num)

        ret,frame = video.read() 
        
        if ret == False:
            break

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #original in bgr. need to convert to rgb

        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 1) #face detector for cropping
        #cv2.imshow("face",frame)
        

        for idx, det in enumerate(dets): #get box of face detected by dlib
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left() 
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            if conf > 1.0: #if face detected, crop image
                bbox_width = abs(x_max - x_min)#get width of face
                bbox_height = abs(y_max - y_min)#get height of face
                x_min -= 2 * bbox_width // 4 #calculate values for cropping
                x_max += 2 * bbox_width // 4
                y_min -= 3 * bbox_height // 4
                y_max += bbox_height // 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[y_min:y_max,x_min:x_max]
                img = Image.fromarray(img)

                # Transform
                img = transformations(img) #declared above
                img_shape = img.size() #get shape of image
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)#use cuda

                yaw, pitch, roll = model(img)#evaluate ypr

                yaw_predicted = F.softmax(yaw)# get predicted value. 3 fully connected layer to evaluate indepedently
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data * idx_tensor) * 3 - 99

                # Print new frame with cube and axis
                txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                
                # Plot expanded bounding box
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                #cv2.waitKey(0)
                cv2.putText(frame, str(yaw_predicted), (00,250), cv2.FONT_HERSHEY_SIMPLEX , 1,  
                 (0, 0, 255) , 2, cv2.LINE_AA, False)
                print (yaw_predicted)
                if yaw_predicted>48 or yaw_predicted<-48:
                    cv2.putText(frame, "HEAD IS NOT FACING STRAIGHT", (00,185), cv2.FONT_HERSHEY_SIMPLEX , 1,  
                 (0, 0, 255) , 2, cv2.LINE_AA, False)
               
                else:
                    cv2.putText(frame, "Normal", (00,185), cv2.FONT_HERSHEY_SIMPLEX , 1,  
                 (0, 0, 255) , 2, cv2.LINE_AA, False)
                    
                
                    
                
                
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            #headpose.t.summary()
            break
                  
        #out.write(frame)
        
        frame_num += 1
    #cv2.waitKey(0)
    out.release()
    video.release()