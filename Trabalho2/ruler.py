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


def undistort_frame(imgc, mtx, dist):
    """
    Undistorts the frame given mtx and dist.
    """
    # finds a refined camera matrix based on a free scaling parameter alpha and the distortions
    h, w = imgc.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    
    # undistort and crop the image
    undist = cv2.undistort(imgc, mtx, dist, None, new_camera_mtx)
    x, y, w, h = roi
    undist = undist[y:y+h, x:x+w]
    
    # returns the undistorted image
    return undist


def display(mode, fpath, mtx, dist, rt):
    """
    Shows an image chosen by the user.
    Also shows the undistorted windows given the calibration parameters.
    Allows measuring of distance in world coordinates.
    """
    # inform the controls
    print('Left-click to select the pixels. Right-click to clear selections.')
    
    # number of chessboard corners
    chess_dim = (8, 6)
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # load the pattern size
    measures = {}
    with open('pattern_size.txt', 'r') as pattern_file:
        for line in pattern_file:
            measures['pattern'] = float(line)

    # load image and make a copy
    img = [None, None]
    img[0] = cv2.imread(fpath)
    img[0] = resize_custom(img[0], 'qhd', verbose=False)
    img[1] = undistort_frame(img[0], mtx, dist)
    print('Image size: {}x{}\n'.format(img[0].shape[1], img[0].shape[0]))
    
    # for each image (raw, undistorted)
    for i in range(0, 2):
    
        # find corners in the pattern
        gray = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(gray, chess_dim, None)
        if retval is True:
            
            # refine the corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # project axis on image
            axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1, 3) * measures['pattern']
            rmat, tvec = rt[:, 0:3], rt[:, 3]
            rvec, _ = cv2.Rodrigues(rmat)
            img_points, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
            
            # draw axis on image
            corner = tuple(corners2[0].ravel())
            img[i] = cv2.line(img[i], corner, tuple(img_points[0].ravel()), (255, 0, 0), 2)
            img[i] = cv2.line(img[i], corner, tuple(img_points[1].ravel()), (0, 255, 0), 2)
            img[i] = cv2.line(img[i], corner, tuple(img_points[2].ravel()), (0, 0, 255), 2)
        
        else:
            print('failed?')
    
    # dictionary with event flags and mouse pos
    cb_params_raw = {'left_click': False, 'right_click': False, 'mid_click': False, 'x': 0, 'y': 0}
    cb_params_und = {'left_click': False, 'right_click': False, 'mid_click': False, 'x': 0, 'y': 0}
    
    # configure windows and mouse callback function
    cv2.namedWindow('raw')
    cv2.setMouseCallback('raw', mouse_callback, cb_params_raw)
    cv2.namedWindow('undistorted')
    cv2.setMouseCallback('undistorted', mouse_callback, cb_params_und)
    
    # which mouse click we are at
    mouseClick = [0, 0]    
    
    # pixel positions
    pos = [[None, None], [None, None]]
    
    # world positions
    world_pos = [[None, None], [None, None]]
    
    # projection matrix and its components
    p = np.matmul(mtx, rt)
    
    # copy images
    imgc = [None, None]
    imgc[0] = img[0].copy()
    imgc[1] = img[1].copy()
    
    # keep looping until 'q' is pressed
    while True:
        
        # select cb_params for monitoring
        cb_params_curr = [cb_params_raw, cb_params_und]
        
        # check for mouse clicks in the windows
        for i, cb_params in enumerate(cb_params_curr):
        
            # determine the window name
            wn = '[Raw] ' if i == 0 else '[Undistorted] '
        
            # if the left mouse button is clicked...
            if cb_params['left_click']:
                
                # if it was the first click, mark the pixel
                if mouseClick[i] == 0:
                    pos[i][1] = None
                    pos[i][0] = (cb_params['x'], cb_params['y'])
                    imgc[i][pos[i][0][1]-1:pos[i][0][1]+1, pos[i][0][0]-1:pos[i][0][0]+1, :] = [0, 0, 255]
                    print('{} First pixel selected: {}'.format(wn, pos[i][0]))
                    
                    # get world coordinate
                    img_point = np.asarray([pos[i][0][0], pos[i][0][1], 1])
                    world_pos[i][0], _, _, _ = np.linalg.lstsq(p, img_point)
                    world_pos[i][0] /= world_pos[i][0][3]
                    print('{} World coordinate: {}'.format(wn, world_pos[i][0]))
                    mouseClick[i] = 1
                    
                # if it was the second click, mark the pixel and draw the line
                elif mouseClick[i] == 1:
                    pos[i][1] = (cb_params['x'], cb_params['y'])
                    imgc[i][pos[i][1][1]-1:pos[i][1][1]+1, pos[i][1][0]-1:pos[i][1][0]+1, :] = [0, 0, 255]
                    print('{} Second pixel selected: {}'.format(wn, pos[i][1]))
                    
                    # get world coordinate
                    img_point = np.asarray([pos[i][1][0], pos[i][1][1], 1])
                    world_pos[i][1], _, _, _ = np.linalg.lstsq(p, img_point)
                    world_pos[i][1] /= world_pos[i][1][3]
                    print('{} World coordinate: {}'.format(wn, world_pos[i][1]))
                    mouseClick[i] = 1
                    
                    # get the pixel distance
                    imgc[i] = cv2.line(imgc[i], pos[i][0], pos[i][1], (0, 0, 255))
                    distance = np.linalg.norm(np.asarray(pos[i][1]) - np.asarray(pos[i][0]))
                    print('{} Pixel distance: {:f}\n'.format(wn, distance))
                    
                    # get the world distance
                    real_dist = np.linalg.norm(np.asarray(world_pos[i][1]) - np.asarray(world_pos[i][0]))                    
                    print('{} Real distance: {:f}\n'.format(wn, real_dist))
                    
                    mouseClick[i] = 0
                    
                # clear the click flag
                cb_params['left_click'] = False
                
            # if the right mouse button is clicked, clear all selections
            if cb_params['right_click']:
                imgc[i][:, :, :] = img[i][:, :, :]
                print('\n{} All selections cleared\n'.format(wn))
                mouseClick[i] = 0
                cb_params['right_click'] = False
            
            # update the image
            wn = 'raw' if i == 0 else 'undistorted'
            cv2.imshow(wn, imgc[i])
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    # finish up
    cv2.destroyAllWindows()