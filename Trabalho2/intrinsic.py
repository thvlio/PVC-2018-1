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


def run_intrinsic_calibration(intset_folder, xml_folder):
    """
    Executes the intrinsic calibration procedure using the images stored in images/intset/.
    The procedure will be executed 5 times, with images from each setn/ folder, and the .xml files will be saved to xml/.
    """
    # number of chessboard corners
    chess_dim = (8, 6)
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # list of matrices and distortions
    mtxs = []
    dists = []
    
    # run the calibration procedure 5 times
    print('Starting intrinsic calibration...')
    for i in range(0, 5):
    
        # prepare object points array
        obj_point = np.zeros((chess_dim[0]*chess_dim[1], 3), np.float32)
        obj_point[:, :2] = np.mgrid[0:chess_dim[0], 0:chess_dim[1]].T.reshape(-1, 2)
        
        # arrays to store object points and image points from all images
        obj_points = [] # 3d point in world coordinates
        img_points = [] # 2d point in image coordinates
        
        # get images from set folder
        set_path = os.path.join(intset_folder, 'set{}'.format(i+1))
        fnames = os.listdir(set_path)
        
        # for all images in the folder...
        for fname in fnames:
            
            # read image and convert it to grayscale
            img = cv2.imread(os.path.join(set_path, fname))
            img = resize_custom(img, 'qhd', verbose=False)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # find the chessboard corners
            retval, corners = cv2.findChessboardCorners(gray, chess_dim, None)
            
            # if found, refine the image and object points and add them to the list
            if retval is True:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(obj_point)
                img_points.append(corners2)
            else:
                print('failed?')
                
        # find the camera matrix and the distortion coefficients
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        mtxs.append(mtx)
        dists.append(dist)
                
        # print message
        print('Progress: {}/5'.format(i+1))
    
    # calculates mean and standard deviation for the camera matrix and the distortion coefficients
    #print('mtxs\n', mtxs)
    #print('dists\n', dists)
    mtx_mean = np.mean(mtxs, axis=0)
    mtx_dev = np.std(mtxs, axis=0)
    dist_mean = np.mean(dists, axis=0)
    dist_dev = np.std(dists, axis=0)
    
    # print the results
    # print('mtx_mean: {}\n'.format(mtx_mean))
    # print('mtx_dev: {}\n'.format(mtx_dev))
    # print('dist_mean: {}\n'.format(dist_mean))
    # print('dist_dev: {}\n'.format(dist_dev))
    
    # calculates the re-projection error
    # mean_error = 0
    # for i in range(len(obj_points)):
    #     img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    #     error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2)/len(img_points2)
    #     mean_error += error
    # total_error = mean_error/len(obj_points)
    # print('total error: {}'.format(total_error)
        
    # saves the data to xml files
    print('Saving .xml files...')
    mtx_file = cv2.FileStorage(os.path.join(xml_folder, 'intrinsics.xml'), cv2.FILE_STORAGE_WRITE)
    mtx_file.write('camera_matrix', mtx_mean)
    mtx_file.release()
    dist_file = cv2.FileStorage(os.path.join(xml_folder, 'distortions.xml'), cv2.FILE_STORAGE_WRITE)
    dist_file.write('distortion_coefficients', dist)
    dist_file.release()
    
    # returns the data
    return mtx_mean, dist_mean


def load_intrinsic_parameters(xml_folder):
    """
    Loads intrinsic parameters from the .xml files in the given xml folder.
    """
    # read the data from the files
    print('Loading intrinsic .xml files...')
    mtx_file = cv2.FileStorage(os.path.join(xml_folder, 'intrinsics.xml'), cv2.FILE_STORAGE_READ)
    mtx = mtx_file.getNode('camera_matrix').mat()
    mtx_file.release()
    dist_file = cv2.FileStorage(os.path.join(xml_folder, 'distortions.xml'), cv2.FILE_STORAGE_READ)
    dist = dist_file.getNode('distortion_coefficients').mat()
    dist_file.release()
    #print(mtx, dist)
    # return the parameters
    return mtx, dist
    
    
def display(mode, fpath, mtx, dist):
    """
    Shows an image, a video or a webcam stream depending on the mode parameter.
    Also shows the undistorted windows given the calibration parameters.
    """
    # inform the controls
    print('Left-click to select the pixels. Right-click to clear selections.')
    
    # load video capture object if in video or webcam mode
    if mode == 'video' or mode == 'webcam':
        video_cap = cv2.VideoCapture(fpath)
        
    # load image and make a copy if in image mode. also undistort it
    elif mode == 'image':
        img = [None, None]
        img[0] = cv2.imread(fpath)
        img[0] = resize_custom(img[0], 'qhd', verbose=False)
        img[1] = undistort_frame(img[0], mtx, dist)
        print('Image size: {}x{}\n'.format(img[0].shape[1], img[0].shape[0]))
    
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
    
    # init the imgc list
    imgc = [None, None]
    
    # if image mode, copy image
    if mode == 'image':
        imgc[0] = img[0].copy()
        imgc[1] = img[1].copy()
    
    # keep looping until 'q' is pressed
    while True:
        
        # select cb_params for monitoring
        cb_params_curr = [cb_params_raw, cb_params_und]
        
        # grab frames if in video or webcam mode
        if mode == 'video' or mode == 'webcam':
            grabbed, imgc[0] = video_cap.read()
            imgc[0] = resize_custom(imgc[0], 'qhd', verbose=False)
            imgc[1] = undistort_frame(imgc[0], mtx, dist)
            
            # if no frame was grabbed, continue until a frame is grabbed
            # if it fails too many times, end the program
            if not grabbed:
                num_fails += 1
                if (num_fails < 10):
                    continue
                else:
                    break
        
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
                    if mode == 'image':
                        imgc[i][pos[i][0][1]-1:pos[i][0][1]+1, pos[i][0][0]-1:pos[i][0][0]+1, :] = [0, 0, 255]
                    print('{} First pixel selected: {}'.format(wn, pos[i][0]))
                    mouseClick[i] = 1
                    
                # if it was the second click, mark the pixel and draw the line
                elif mouseClick[i] == 1:
                    pos[i][1] = (cb_params['x'], cb_params['y'])
                    if mode == 'image':
                        imgc[i][pos[i][1][1]-1:pos[i][1][1]+1, pos[i][1][0]-1:pos[i][1][0]+1, :] = [0, 0, 255]
                    print('{} Second pixel selected: {}'.format(wn, pos[i][1]))
                    if mode == 'image':
                        imgc[i] = cv2.line(imgc[i], pos[i][0], pos[i][1], (0, 0, 255))
                    distance = np.linalg.norm(np.asarray(pos[i][1]) - np.asarray(pos[i][0]))
                    print('{} Distance: {:f}\n'.format(wn, distance))
                    mouseClick[i] = 0
                    
                # clear the click flag
                cb_params['left_click'] = False
                
            # if the right mouse button is clicked, clear all selections
            if cb_params['right_click']:
                if mode == 'image':
                    imgc[i][:, :, :] = img[i][:, :, :]
                else:
                    pos[i] = [None, None]
                print('\n{} All selections cleared\n'.format(wn))
                mouseClick[i] = 0
                cb_params['right_click'] = False
        
            # if in video or webcam mode, keep drawing the line on the frame
            if (mode == 'video' or mode == 'webcam'):
                if pos[i][0] is not None:
                    imgc[i][pos[i][0][1]-1:pos[i][0][1]+1, pos[i][0][0]-1:pos[i][0][0]+1, :] = [0, 0, 255]
                if pos[i][1] is not None:
                    imgc[i][pos[i][1][1]-1:pos[i][1][1]+1, pos[i][1][0]-1:pos[i][1][0]+1, :] = [0, 0, 255]
                if pos[i][0] is not None and pos[i][1] is not None:
                    imgc[i] = cv2.line(imgc[i], pos[i][0], pos[i][1], (0, 0, 255))
            
            # update the image
            wn = 'raw' if i == 0 else 'undistorted'
            cv2.imshow(wn, imgc[i])
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    # finish up
    cv2.destroyAllWindows()