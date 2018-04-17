# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# options for image resizing
STANDARDS = {'qhd': (540, 960), 'hd': (720, 1280), 'fhd': (1080, 1920)}


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


def resize_custom(args, img, verbose=True):
    """
    Resizes the image or frame to be smaller than or equal in size to the chosen standard while maintaining the aspect ratio.
    Can be verbose or not.
    """
    # get the chosen standard shape
    shape = STANDARDS[args['resize']]
    
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
        
    print('Image size: {}x{}'.format(img.shape[1], img.shape[0]))


def get_pixel_info(imgc, x, y, grayscale):
    """
    From the (x, y) coordinates given, retrieves the pixel and show it's info.
    """
    pixel = imgc[y, x, :].copy()
    if grayscale:
        info = '(x, y): ({}, {})\tI: {}'.format(x, y, pixel[0])
    else:
        info = '(x, y): ({}, {})\tRGB: {}'.format(x, y, pixel[::-1])
    print(info)
    return pixel


def highlight_pixels(pixel, imgc, grayscale):
    """
    Receives a pixel retrieved from the image and hightlighs other pixels similar in tone in red.
    If the image is grayscale, the distance between images is calculated in the 1-d space with numpy.absolute.
    If the image is RGB, the distance between images is calculated in the 3-d space with numpy.linalg.norm.
    """
    if grayscale:
        dist = np.abs(imgc[:, :, 0].astype(int) - pixel[0].astype(int))
    else:
        dist = np.linalg.norm(imgc.astype(np.int16) - pixel.astype(np.int16), axis=2)
        
    imgc[dist < 13] = [0, 0, 255] # this piece of black magic is almost 10x faster than the loops below
    
    #for i in np.arange(imgc.shape[0]):
    #    for j in np.arange(imgc.shape[1]):
    #        if dist[i, j] < 13:
    #            imgc[i, j, :] = [0, 0, 255]


def run_image_mode(args):
    """
    Executes the program in image mode, in which an image is opened and can be interacted with.
    """
    # inform the controls
    print('Left-click on a pixel to select it and highlight similar pixels in red. Right-click to reset to original state. Middle-click to save a copy on disk.')
    
    # load image
    img = cv2.imread(args['image'])
    
    # resize if image is larger than the standard selected
    if args['resize'] is not None:
        img = resize_custom(args, img)
        print('Image size: {}x{}'.format(img.shape[1], img.shape[0]))
        
    # check if the image is grayscale
    grayscale = True if np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2]) else False
    if grayscale:
        print('Image is grayscale.')
    
    # copy image and use the copy instead
    imgc = img.copy()
    
    # dictionary with event flags and mouse pos
    cb_params = {'left_click': False, 'right_click': False, 'mid_click': False, 'x': 0, 'y': 0}
    
    # configure window and mouse callback function
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback, cb_params)
    
    # keep looping until 'q' is pressed
    while True:
        
        # if the left mouse button is pressed, the pixel is selected, it's info is shown and similar pixels are highlighted in red
        if cb_params['left_click']:
            pixel = get_pixel_info(imgc, cb_params['x'], cb_params['y'], grayscale)
            highlight_pixels(pixel, imgc, grayscale)
            cb_params['left_click'] = False
        
        # if the right mouse button is clicked, restore the original image
        if cb_params['right_click']:
            imgc[:,:,:] = img[:,:,:]
            print('Image reset.')
            cb_params['right_click'] = False
            
        # if the middle mouse button is clicked, save a copy of the image
        if cb_params['mid_click']:
            cp_name = args['image'][0:-4] + '_copy.jpg'
            cv2.imwrite(cp_name, imgc)
            print('Image copy saved with name {}'.format(cp_name))
            cb_params['mid_click'] = False
        
        cv2.imshow('image', imgc)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    # finish up
    cv2.destroyAllWindows()


def run_video_mode(args):
    """
    Executes the program in video mode, in which a video is opened and can be interacted with.
    """
    # inform the controls
    print('Left-click on a pixel to select it and highlight similar pixels in red. Right-click to reset to original state. Middle-click to save a copy on disk.')
    
    # load video capture object
    video_cap = cv2.VideoCapture(args['video'])
    
    # get video original FPS
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    period = 1.0/fps
    
    # dictionary with event flags and mouse pos
    cb_params = {'left_click': False, 'right_click': False, 'mid_click': False, 'x': 0, 'y': 0}
    
    # configure window and mouse callback function
    cv2.namedWindow('video')
    cv2.setMouseCallback('video', mouse_callback, cb_params)
    
    # the scope for these objects should be outside the loop
    pixel = np.zeros(3)
    pixel_selected = False
    
    # keep looping until 'q' is pressed
    while True:
        
        # counting time
        start = time.time()
        
        # grab and retrieve the frame
        grabbed, frame = video_cap.read()
        
        # if there are no more frames left, finish the execution
        if not grabbed:
            print('Reached end of video.')
            break
        
        # resize if frame is larger than the standard selected
        if args['resize'] is not None:
            frame = resize_custom(args, frame, verbose=False)
            
        # check if the frame is grayscale
        # grayscale = True if np.array_equal(frame[:,:,0], frame[:,:,1]) and np.array_equal(frame[:,:,1], frame[:,:,2]) else False
        # if grayscale:
        #     print('Frame is grayscale.')
        grayscale = False
        
        # if the left mouse button is pressed, the pixel is selected, it's info is shown and similar pixels are highlighted in red
        if cb_params['left_click']:
            pixel = get_pixel_info(frame, cb_params['x'], cb_params['y'], grayscale)
            cb_params['left_click'] = False
            pixel_selected = True
            
        if pixel_selected:
            highlight_pixels(pixel, frame, grayscale)
        
        # if the right mouse button is clicked, stop highlighting colors
        if cb_params['right_click']:
            print('Highlighting reset.')
            cb_params['right_click'] = False
            pixel_selected = False
        
        # if the middle mouse button is clicked, save a copy of the current frame
        if cb_params['mid_click']:
            cp_name = args['video'][0:-4] + '_' + str(time.time()) + '_frame_copy.jpg'
            cv2.imwrite(cp_name, frame)
            print('Frame copy saved with name {}'.format(cp_name))
            cb_params['mid_click'] = False
        
        cv2.imshow('video', frame)
        
        # cap the fps to the original fps (approximate)
        diff = int ((period - time.time() + start) * 1000)
        key = cv2.waitKey(diff) & 0xFF if diff > 0 else cv2.waitKey(1) & 0xFF  
        
        if key == ord('q'):
            break
    
    # finish up
    cv2.destroyAllWindows()


def run_webcam_mode(args):
    """
    Executes the program in webcam mode, in which frames are read from the webcam and can be interacted with.
    """
    # inform the controls
    print('Left-click on a pixel to select it and highlight similar pixels in red. Right-click to reset to original state. Middle-click to save a copy on disk.')
    
    # load video capture object
    video_cap = cv2.VideoCapture(args['webcam'])
    
    # dictionary with event flags and mouse pos
    cb_params = {'left_click': False, 'right_click': False, 'mid_click': False, 'x': 0, 'y': 0}
    
    # configure window and mouse callback function
    cv2.namedWindow('webcam')
    cv2.setMouseCallback('webcam', mouse_callback, cb_params)
    
    # the scope for these objects should be outside the loop
    pixel = np.zeros(3)
    pixel_selected = False
    num_fails = 0
    
    # keep looping until 'q' is pressed
    while True:
        
        # grab and retrieve the frame
        grabbed, frame = video_cap.read()
        
        # if no frame was grabbed, continue until a frame is grabbed
        # if it fails too many times, end the program
        if not grabbed:
            num_fails += 1
            if (num_fails < 10):
                continue
            else:
                break
        
        # resize if frame is larger than the standard selected
        if args['resize'] is not None:
            frame = resize_custom(args, frame, verbose=False)
            
        # check if the frame is grayscale
        # grayscale = True if np.array_equal(frame[:,:,0], frame[:,:,1]) and np.array_equal(frame[:,:,1], frame[:,:,2]) else False
        # if grayscale:
        #     print('Frame is grayscale.')
        grayscale = False
        
        # if the left mouse button is pressed, the pixel is selected, it's info is shown and similar pixels are highlighted in red
        if cb_params['left_click']:
            pixel = get_pixel_info(frame, cb_params['x'], cb_params['y'], grayscale)
            cb_params['left_click'] = False
            pixel_selected = True
            
        if pixel_selected:
            highlight_pixels(pixel, frame, grayscale)
        
        # if the right mouse button is clicked, stop highlighting colors
        if cb_params['right_click']:
            print('Highlighting reset.')
            cb_params['right_click'] = False
            pixel_selected = False
            
        # if the middle mouse button is clicked, save a copy of the current frame
        if cb_params['mid_click']:
            cp_name = 'webcam_' + str(time.time()) + '_frame_copy.jpg'
            cv2.imwrite(cp_name, frame)
            print('Frame copy saved with name {}'.format(cp_name))
            cb_params['mid_click'] = False
        
        cv2.imshow('webcam', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # finish up
    cv2.destroyAllWindows()


def main():
    
    # setup argument parser
    ap = argparse.ArgumentParser('Select a mode and if applicable give a path to either an image or a video or a device number.')
    ap.add_argument('-i', '--image', type=str,
        help='Use image files in the program, must be followed by path to the image to be used')
    ap.add_argument('-v', '--video', type=str,
        help='Use video files in the program, must be followed by path to the video to be used')
    ap.add_argument('-w', '--webcam', type=int, const=0, nargs='?',
        help='Use webcam instead of images or videos in the program, can optionally followed by device number')
    ap.add_argument('-r', '--resize', type=str,
        help='Resize image to the selected quality standard (qhd, hd or fhd) if the image dimensions surpass it (maintains aspect ratio)')
    args = vars(ap.parse_args())

    # check for mode errors
    mode = None
    print(args)
    modes_list = [False if args['image'] is None else True, False if args['video'] is None else True, False if args['webcam'] is None else True]
    if True not in modes_list:
        print('No mode was selected. You can select either image mode, video mode or webcam mode. Use --help to check usage of the program.')
    elif modes_list.count(True) > 1:
        print('Only one mode can be selected. Select either image mode, video mode or webcam mode. Use --help to check usage of the program.')
    else:
        mode = ['image', 'video', 'webcam'][modes_list.index(True)]
        print('Selected ' + mode + ' mode.')
        if (mode == 'image' or mode == 'video'):
            print('File to be used: {}'.format(args[mode]))
            
    # check for file existence and webcam device
    if (mode == 'image' or mode == 'video') and not os.path.isfile(args[mode]):
        print('File {} not found.'.format(args[mode]))
        mode = None
    if mode == 'webcam':
        print('Camera device selected: {}'.format(args[mode]))
    
    # check if the resize option is implemented
    if args['resize'] is not None and args['resize'] not in STANDARDS:
        print('The selected quality stardard ({}) is not available. The options are qhd, hd and fhd.'.format(args['resize']))

    # run selected mode
    if mode == 'image':
        run_image_mode(args)
    
    elif mode == 'video':
        run_video_mode(args)
        
    elif mode == 'webcam':
        run_webcam_mode(args)


if __name__ == '__main__':
    main()