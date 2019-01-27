# import the necessary packages
import numpy as np
import cv2
import os
import time

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


def run_extrinsic_calibration(mtx, dist):
    """
    Executes the extrinsic calibration procedure using the images stored in imgs/cal.
    """
    # list of extrinsic matrices
    rts = []
    
    # number of chessboard corners
    chess_dim = (8, 6)
    num_points = chess_dim[0]*chess_dim[1]
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # run the calibration procedure 3 times
    print('Starting extrinsic calibration...')
    
    # prepare object points array
    obj_point = np.zeros((num_points, 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:chess_dim[0], 0:chess_dim[1]].T.reshape(-1, 2) * 28
    
    # for all images in the folder...
    for fname in ['imgs/calL.png', 'imgs/calR.png']:
        
        # read image and convert it to grayscale
        img = cv2.imread(fname)
        # img = resize_custom(img, 'fhd', verbose=False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # find the chessboard corners
        retval, corners = cv2.findChessboardCorners(gray, chess_dim, None)
        
        # if found, refine the corners and add object points and image points to their lists
        if retval is True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            #corners2 = corners2.reshape(-1, corners2.shape[-1])
            #obj_points[i*num_points:(i+1)*num_points, :] = obj_point
            #img_points[i*num_points:(i+1)*num_points, :] = corners2
        else:
            print('failed?')
            
        # get the extrinsic parameters (rotation and traslation vectors)
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_point, corners2, mtx, dist, iterationsCount=200)
        rmat, _ = cv2.Rodrigues(rvec)
        rt = np.concatenate((rmat, tvec), axis=1)
        rts.append(rt)
        
        # project axis on image
        # axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1, 3) * 28
        # rmat, tvec = rt[:, 0:3], rt[:, 3]
        # rvec, _ = cv2.Rodrigues(rmat)
        # img_points, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        
        # draw axis on image
        # corner = tuple(corners2[0].ravel())
        # img = cv2.line(img, corner, tuple(img_points[0].ravel()), (255, 0, 0), 5)
        # img = cv2.line(img, corner, tuple(img_points[1].ravel()), (0, 255, 0), 5)
        # img = cv2.line(img, corner, tuple(img_points[2].ravel()), (0, 0, 255), 5)
        # img[:,:,:] = img[:,:,:]
        
        # show the image
        # cv2.imshow('axis', img)
        # cv2.waitKey(0)
        
    return rts[0], rts[1]


def resize_custom(img, std, c_shape=None, verbose=False):
    """
    Resizes the image or frame to be smaller than or equal in size to the chosen standard while maintaining the aspect ratio.
    Can be verbose or not.
    """
    # get the chosen standard shape
    if std == 'custom':
        shape = c_shape
    else:
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


def drawlines(img1, img2, lines, pts1, pts2):
    """
    Draw the epilines for points in img2 on img1.
    """
    r, c = img1.shape[:2]
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def find_correspondences(imgL, imgR, desc, show_matches):
    """
    Determines the pairs of corresponding points in the images.
    """
    # split the descriptor name
    desc_name, n_features = desc.split('-')
    n_features = int(n_features)
    
    # use the ORB descriptor
    if desc_name == 'orb':
    
        # initialize the ORB descriptor, then find and descript the keypoints
        orb = cv2.ORB_create(n_features)
        kpL, desL = orb.detectAndCompute(imgL, None)
        kpR, desR = orb.detectAndCompute(imgR, None)
        
        # create BFMatcher object, match descriptors and sort them
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desL, desR)
        matches = sorted(matches, key = lambda x:x.distance)
        
        ptsL = []
        ptsR = []
        
        # save the points
        for m in matches:
            ptsL.append(kpL[m.queryIdx].pt)
            ptsR.append(kpR[m.trainIdx].pt)
    
    # use the SIFT descriptor
    elif desc_name == 'sift':

        # initialize the SIFT descriptor, the find and descript the keypoints
        sift = cv2.xfeatures2d.SIFT_create(n_features)
        kpL, desL = sift.detectAndCompute(imgL, None)
        kpR, desR = sift.detectAndCompute(imgR, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        # find matches with FLANN
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(desL, desR, k=2)

        matches = []
        ptsL = []
        ptsR = []

        # ratio test as per Lowe's paper
        for m, n in raw_matches:
            if m.distance < 0.5 * n.distance:
                matches.append(m)
                ptsL.append(kpL[m.queryIdx].pt)
                ptsR.append(kpR[m.trainIdx].pt)
    
    # draw the matches
    print("{} matching keypoints found.".format(len(matches)))
    if show_matches:
        imgM = cv2.drawMatches(imgL, kpL, imgR, kpR, matches, None, flags=2)            
        cv2.imshow('matches', resize_custom(imgM, 'hd'))
        cv2.waitKey(0)
    
    # find the fundamental matrix
    ptsL = np.int32(ptsL)
    ptsR = np.int32(ptsR)
    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_LMEDS)

    # print('F: {}\n'.format(F))
    # print('Mask: {}\n'.format(F))

    # We select only inlier points
    ptsL = ptsL[mask.ravel()==1]
    ptsR = ptsR[mask.ravel()==1]
    inlier_matches = []
    for i, v in enumerate(mask.ravel()):
        if v == 1:
            inlier_matches.append(matches[i])

    # draw the inlier matches
    print("{} inlier matching keypoints found.".format(len(inlier_matches)))
    if show_matches:
        imgM = cv2.drawMatches(imgL, kpL, imgR, kpR, inlier_matches, None, flags=2)            
        cv2.imshow('matches', resize_custom(imgM, 'hd'))
        cv2.waitKey(0)
    
    ##
    ## remove this code later, there is no need to show the lines except for debugging
    ##
    if show_matches:
        
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(ptsR.reshape(-1,1,2), 2, F)
        lines1 = lines1.reshape(-1,3)
        img5, img6 = drawlines(imgL, imgR, lines1, ptsL, ptsR)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(ptsL.reshape(-1,1,2), 1, F)
        lines2 = lines2.reshape(-1,3)
        img3, img4 = drawlines(imgR, imgL, lines2, ptsR, ptsL)
        
        # show a montage of the fours images
        img = np.vstack((np.hstack((img3, img4)), np.hstack((img5, img6))))
        img = resize_custom(img, 'hd')
        cv2.imshow('image', img)
        cv2.waitKey(0)
        
    # return the fundamental matrix and the matching points
    return F, ptsL, ptsR


def compute_stereo_params(imgL, imgR, F, ptsL, ptsR):
    """
    Compute the focal lenght and the baseline for the stereo rig.
    """
    # compute the rectification transform for the images
    _, H1, H2 = cv2.stereoRectifyUncalibrated(ptsL, ptsR, F, (imgL.shape[1], imgL.shape[0]))
    
    # warp the images
    warped1 = cv2.warpPerspective(imgL, H1, (imgL.shape[1], imgL.shape[0]))
    warped2 = cv2.warpPerspective(imgR, H2, (imgR.shape[1], imgR.shape[0]))
        
    # cv2.imshow('imgL_rec', warped1)
    # cv2.waitKey(0)
    # cv2.imshow('imgR_rec', warped2)
    # cv2.waitKey(0)
    
    # load the intrinsic calibration matrix and get the values from it
    mtx, dist = load_intrinsic_parameters('xml')
    f = (mtx[0, 0] + mtx[1, 1])/2
    
    # run extrinsic calibration to find the baseline
    dist[:] = 0.0
    rt1, rt2 = run_extrinsic_calibration(mtx, dist)
    (r1, t1) = (rt1[0:3, 0:3], rt1[0:3, 3])
    (r2, t2) = (rt2[0:3, 0:3], rt2[0:3, 3])
    b = np.linalg.norm(np.matmul(-r1.T, t1) + np.matmul(r2.T, t2))
    
    # return some useful data
    return warped1, warped2, f, b, H1, H2


def compute_disparity_and_depth(imgL, imgR, scale, window_size, min_disp, max_disp, tol, f, b, show_maps, d_map):
    """
    Compute the disparity and depth map from the fundamental matrix and the matching points.
    """
    # disparity range of values (based on the size of the room)
    dmax_limit = max_disp # if close to the camera
    dmin_limit = min_disp # if at the opposite wall
    
    # image size
    (h, w) = imgL.shape[:2]
    
    # horizontal searching image for the template matching
    limit = w // 8 # 8 is a good value for all pictures
    
    # window size for template matching (must be odd)
    ws = int(window_size * scale) # 64 for aloe and baby (128 for the phone pictures)
    window_size = ws+1 if ws % 2 == 0 else ws
    pad = (window_size - 1) // 2
    
    # if no map was given
    if d_map is None:
    
        # empty disparity map
        disp_map = np.zeros((int(h*scale), int(w*scale), 1))
        
        # for all values of window center in y
        for yc in np.arange(pad, h-pad, int(1/scale)):
            
            # show progress
            comp = int(100 * ((yc-pad) / (h-pad)))
            miss = 100 - comp
            print("Progress: [{}{}]".format('#'*comp, ' '*miss))
            
            # for all values of window center in x
            for xc in np.arange(pad, w-pad, int(1/scale)):
                
                # define the template as a window around the center pixel
                template = imgL[yc-pad:yc+pad+1, xc-pad:xc+pad+1, :]
                
                # define limits
                xlim_l = max(xc-limit, 0)
                xlim_h = min(xc+limit+1, w)
                
                # define the horizontal searching slice
                h_slice = imgR[yc-pad:yc+pad+1, xlim_l:xlim_h, :]
                
                # match the template and get the center
                # mth = {'ccoeff': cv2.TM_CCOEFF_NORMED, 'ccorr': cv2.TM_CCORR_NORMED, 'sqdiff': cv2.TM_SQDIFF_NORMED}
                res = cv2.matchTemplate(h_slice, template, cv2.TM_CCORR_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                # display the points being mapped
                # img = np.hstack((imgL, imgR))
                # tl = (max_loc[0] + xlim_l-pad + w, yc-pad)
                # br = (tl[0] + template.shape[1], tl[1] + template.shape[0])
                # cv2.rectangle(img, (xc-pad, yc-pad), (xc+pad, yc+pad), 255, 2)
                # cv2.rectangle(img, tl, br, 255, 2)
                # cv2.imshow('img', resize_custom(img, 'qhd'))
                # cv2.waitKey(1)
                
                # get the disparity
                # if method == 'ccoeff':
                #     if max_val > 0.6:
                #         disp = xc - max_loc[0] - xlim_l
                #     else:
                #         disp = np.nan
                # elif method == 'ccorr':
                #     disp = xc - max_loc[0] - xlim_l
                # elif method == 'sqdiff':
                #     disp = xc - min_loc[0] - xlim_l
                disp = xc - max_loc[0] - xlim_l
                disp_map[int((yc-pad)*scale), int((xc-pad)*scale)] = disp
        
        # clamp the values
        disp_map[disp_map < dmin_limit] = dmin_limit
        disp_map[disp_map > dmax_limit] = dmax_limit
        
        # filter the values
        outliers = np.absolute(disp_map - np.mean(disp_map)) / np.std(disp_map)
        disp_map[outliers > tol] = np.nan
    
    # if a map was given
    else:
        disp_map = d_map
    
    # get min and max disparity
    dmin = np.nanmin(disp_map)
    dmax = np.nanmax(disp_map)
    
    # bring the values to the range 0-255
    disp_map_normed = (disp_map - np.nanmin(disp_map)) / (np.nanmax(disp_map) - np.nanmin(disp_map)) * 255
    disp_map_normed = disp_map_normed.astype('uint8')
    
    # show the scaled original image
    if show_maps:
        resized_img = resize_custom(imgL, 'custom', c_shape=disp_map_normed.shape[:2])
        cv2.imshow('original', resized_img)
    
    # show the disparity image
    if show_maps:
        cv2.imshow('disp', disp_map_normed)
    
    # save the image
    # img_name = 'd3/disp_map_n{}_w{}_m{}_t{}.jpg'.format(n, wn, method, int(time.time()-t0))
    # img_name = 'd4/disp_map_{}_{}.jpg'.format(model, mode)
    # cv2.imwrite(img_name, disp_map)
    # print('Wrote: {}'.format(img_name))
    
    # compute the logarithm of the depth map
    depth_map = np.log(b * f / disp_map)
    
    # mask of finite values (not nan and not inf)
    isfin = np.isfinite(depth_map)
    
    # normalize the logarithm of the depth map to the range 0-254
    depth_map_normed = (depth_map - depth_map[isfin].min()) / (depth_map[isfin].max() - depth_map[isfin].min()) * 254 + 1
    
    # treat the special cases (+inf -> 255, -inf -> 0, nan -> 0)
    depth_map_normed[np.isposinf(depth_map_normed)] = 255
    depth_map_normed[np.isneginf(depth_map_normed)] = 1
    depth_map_normed[np.isnan(depth_map_normed)] = 0
    
    # make sure the map is uint8
    depth_map_normed = depth_map_normed.astype('uint8')
    
    # show the depth map
    if show_maps:
        cv2.imshow('depth', depth_map_normed)
        cv2.waitKey(0)
    
    # get min and max depth
    Zmin = b*f/dmax
    Zmax = b*f/dmin
    print('Zmin: {}, Zmax: {}'.format(Zmin, Zmax))
    
    # return the maps
    return disp_map, disp_map_normed, depth_map, depth_map_normed, Zmin, Zmax
    

def display(imgL, imgR, H1, H2, f, b):
    """
    Shows two images from a stereo rig to the user and allows measuring of object size.
    """
    # inform the controls
    print('Left-click to select the pixels. Right-click to clear selections.')
    
    # dictionary with event flags and mouse pos
    cb_params_L = {'left_click': False, 'right_click': False, 'mid_click': False, 'x': 0, 'y': 0}
    cb_params_R = {'left_click': False, 'right_click': False, 'mid_click': False, 'x': 0, 'y': 0}
    
    # configure windows and mouse callback function
    cv2.namedWindow('left')
    cv2.setMouseCallback('left', mouse_callback, cb_params_L)
    cv2.namedWindow('right')
    cv2.setMouseCallback('right', mouse_callback, cb_params_R)
    
    # which mouse click we are at
    mouseClick = [0, 0]    
    
    # pixel positions
    pos = [[None, None], [None, None]]
    
    # world positions
    warped_pos = [[None, None], [None, None]]
    
    # copy images
    H = [H1, H2]
    img = [imgL, imgR]
    imgc = [None, None]
    imgc[0] = img[0].copy()
    imgc[1] = img[1].copy()
    
    # keep looping until 'q' is pressed
    while True:
        
        # select cb_params for monitoring
        cb_params_curr = [cb_params_L, cb_params_R]
        
        # check for mouse clicks in the windows
        for i, cb_params in enumerate(cb_params_curr):
        
            # determine the window name
            wn = '[Left] ' if i == 0 else '[Right] '
        
            # if the left mouse button is clicked...
            if cb_params['left_click']:
                
                # if it was the first click, mark the pixel
                if mouseClick[i] == 0:
                    pos[i][1] = None
                    pos[i][0] = (cb_params['x']*2, cb_params['y']*2)
                    imgc[i][pos[i][0][1]-1:pos[i][0][1]+1, pos[i][0][0]-1:pos[i][0][0]+1, :] = [0, 0, 255]
                    print('{} First pixel selected: {}'.format(wn, pos[i][0]))
                    
                    # get warped coordinate
                    warped_pos[i][1] = None
                    img_point = np.array([[pos[i][0][0], pos[i][0][1]]], dtype='float32')
                    img_point = np.array([img_point])
                    warped_pos[i][0] = cv2.perspectiveTransform(img_point, H[i])[0][0]
                    print('{} Warped coordinate: {}'.format(wn, warped_pos[i][0]))
                    
                    mouseClick[i] += 1
                    
                # if it was the second click, mark the pixel and draw the line
                elif mouseClick[i] == 1:
                    pos[i][1] = (cb_params['x']*2, cb_params['y']*2)
                    imgc[i][pos[i][1][1]-1:pos[i][1][1]+1, pos[i][1][0]-1:pos[i][1][0]+1, :] = [0, 0, 255]
                    print('{} Second pixel selected: {}'.format(wn, pos[i][1]))
                    
                    # draw line
                    cv2.line(imgc[i], pos[i][0], pos[i][1], (0, 0, 255))
                    
                    # get warped coordinate
                    img_point = np.array([[pos[i][1][0], pos[i][1][1]]], dtype='float32')
                    img_point = np.array([img_point])
                    warped_pos[i][1] = cv2.perspectiveTransform(img_point, H[i])[0][0]
                    print('{} Warped coordinate: {}'.format(wn, warped_pos[i][1]))
                    
                    mouseClick[i] = 0
                    
                # clear the click flag
                cb_params['left_click'] = False
                
            # if the right mouse button is clicked, clear all selections
            if cb_params['right_click']:
                imgc[i][:, :, :] = img[i][:, :, :]
                print('\n{} All selections cleared\n'.format(wn))
                mouseClick[i] = 0
                cb_params['right_click'] = False
                warped_pos = [[None, None], [None, None]]
                
            # update the image
            wn = 'left' if i == 0 else 'right'
            cv2.imshow(wn, resize_custom(imgc[i], 'custom', c_shape=(imgc[i].shape[0]//2, imgc[i].shape[1]//2)))
        
        # tell if its ready to compute world coordinate
        not_ready = False
        for l in warped_pos:
            for i in l:
                if i is None:
                    not_ready = True
        
        # if ready
        if not not_ready:
            
            # get the values for the first point
            xl = warped_pos[0][0][0]
            xr = warped_pos[1][0][0]
            yl = warped_pos[0][0][1]
            yr = warped_pos[1][0][1]
            X1 = b*(xl+xr)/(2*(xl-xr))
            Y1 = b*(yl+yr)/(2*(xl-xr))
            Z1 = b*f/(xl-xr)
            P1 = np.array([X1, Y1, Z1])
            
            # get the values for the second point
            xl = warped_pos[0][1][0]
            xr = warped_pos[1][1][0]
            yl = warped_pos[0][1][1]
            yr = warped_pos[1][1][1]
            X2 = b*(xl+xr)/(2*(xl-xr))
            Y2 = b*(yl+yr)/(2*(xl-xr))
            Z2 = b*f/(xl-xr)
            P2 = np.array([X2, Y2, Z2])
            
            # get the distance
            print('f: {}, b: {}, P1: {}, P2: {}'.format(f, b, P1, P2))
            real_dist = np.linalg.norm(P2 - P1)                    
            print('Real distance: {:f}\n'.format(real_dist))
            
            # reset the positions
            warped_pos = [[None, None], [None, None]]
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    # finish up
    cv2.destroyAllWindows()