# import the necessary packages
import numpy as np
import cv2
import os

# import configurations file
import config


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for reading mouse clicks and position.
    If the left mouse button is clicked, the event is flagged and the (x, y) position of the mouse is stored in the parameter dictionary.
    If the right mouse button is clicked, the event is flagged.
    """
    if event == cv2.EVENT_LBUTTONUP:
        param['left_click'] = True
        param['x'] = x
        param['y'] = y
        
    if event == cv2.EVENT_RBUTTONUP:
        param['right_click'] = True
        
    if event == cv2.EVENT_MBUTTONUP:
        param['mid_click'] = True


def resize_custom(img, std, verbose=True):
    """
    Resizes the image or frame to be smaller than or equal in size to the chosen standard while maintaining the aspect ratio.
    Can be verbose or not.
    """
    # get the chosen standard shape
    shape = config.STANDARDS[std]
    
    # if the aspect ratio is greater than the chosen one, the width should be checked
    if img.shape[1]*shape[0] >= img.shape[0]*shape[1] and img.shape[1] > shape[1]:
        if verbose:
            print('Image is too large. Resizing width to {} with the same aspect ratio.'.format(shape[1]))
        return cv2.resize(img, (shape[1], shape[1] * img.shape[0] // img.shape[1]), interpolation=cv2.INTER_AREA)
    
    # if the aspect ratio is less than the chosen one, the height should be checked
    elif img.shape[1]*shape[0] < img.shape[0]*shape[1] and img.shape[0] > shape[0]:
        if verbose:
            print('Image is too large. Resizing height to {} with the same aspect ratio.'.format(shape[0]))
        return cv2.resize(img, (shape[0] * img.shape[1] // img.shape[0], shape[0]), interpolation=cv2.INTER_AREA)
    
    else:
        if verbose:
            print('Image size is smaller than the chosen standard. No need to resize.')
        return img


def display(mode, fpath):
    """
    Shows an image, a video or a webcam stream depending on the mode parameter.
    """
    # inform the controls
    print('Left-click to select the pixels. Right-click to clear selections.')
    
    # load video capture object if in video or webcam mode
    if mode == 'video' or mode == 'webcam':
        video_cap = cv2.VideoCapture(fpath)
        
    # load image and make a copy if in image mode
    elif mode == 'image':
        img = cv2.imread(fpath)
        img = resize_custom(img, 'qhd', verbose=False)
        print('Image size: {}x{}\n'.format(img.shape[1], img.shape[0]))
    
    # dictionary with event flags and mouse pos
    cb_params = {'left_click': False, 'right_click': False, 'mid_click': False, 'x': 0, 'y': 0}
    
    # configure windows and mouse callback function
    cv2.namedWindow('raw')
    cv2.setMouseCallback('raw', mouse_callback, cb_params)
    
    # which mouse click we are at
    mouseClick = 0   
    
    # pixel positions
    pos = [None, None]
    
    # init imgc
    imgc = None
    
    # if image mode, copy image
    if mode == 'image':
        imgc = img.copy()
    
    # keep looping until 'q' is pressed
    while True:
        
        # grab frames if in video or webcam mode
        if mode == 'video' or mode == 'webcam':
            grabbed, imgc = video_cap.read()
            imgc = resize_custom(imgc, 'qhd', verbose=False)
            
            # if no frame was grabbed, continue until a frame is grabbed
            # if it fails too many times, end the program
            if not grabbed:
                num_fails += 1
                if (num_fails < 10):
                    continue
                else:
                    break
    
        # if the left mouse button is clicked...
        if cb_params['left_click']:
            
            # if it was the first click, mark the pixel
            if mouseClick == 0:
                pos[1] = None
                pos[0] = (cb_params['x'], cb_params['y'])
                if mode == 'image':
                    imgc[pos[0][1]-1:pos[0][1]+1, pos[0][0]-1:pos[0][0]+1, :] = [0, 0, 255]
                print('{} First pixel selected: {}'.format('[Raw]', pos[0]))
                mouseClick = 1
                
            # if it was the second click, mark the pixel and draw the line
            elif mouseClick == 1:
                pos[1] = (cb_params['x'], cb_params['y'])
                if mode == 'image':
                    imgc[pos[1][1]-1:pos[1][1]+1, pos[1][0]-1:pos[1][0]+1, :] = [0, 0, 255]
                print('{} Second pixel selected: {}'.format('[Raw]', pos[1]))
                if mode == 'image':
                    imgc = cv2.line(imgc, pos[0], pos[1], (0, 0, 255))
                distance = np.linalg.norm(np.asarray(pos[1]) - np.asarray(pos[0]))
                print('{} Distance: {:f}\n'.format('[Raw]', distance))
                mouseClick = 0
                
            # clear the click flag
            cb_params['left_click'] = False
            
        # if the right mouse button is clicked, clear all selections
        if cb_params['right_click']:
            if mode == 'image':
                imgc[:, :, :] = img[:, :, :]
            else:
                pos = [None, None]
            print('\n{} All selections cleared\n'.format('[Raw]'))
            mouseClick = 0
            cb_params['right_click'] = False
    
        # if in video or webcam mode, keep drawing the line on the frame
        if (mode == 'video' or mode == 'webcam'):
            if pos[0] is not None:
                imgc[pos[0][1]-1:pos[0][1]+1, pos[0][0]-1:pos[0][0]+1, :] = [0, 0, 255]
            if pos[1] is not None:
                imgc[pos[1][1]-1:pos[1][1]+1, pos[1][0]-1:pos[1][0]+1, :] = [0, 0, 255]
            if pos[0] is not None and pos[1] is not None:
                imgc = cv2.line(imgc, pos[0], pos[1], (0, 0, 255))
        
        # update the image
        cv2.imshow('raw', imgc)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    # finish up
    cv2.destroyAllWindows()