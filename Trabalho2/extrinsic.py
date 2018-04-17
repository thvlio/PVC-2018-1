# import the necessary packages
import numpy as np
import cv2
import os

# import configurations file
import config


def load_extrinsic_parameters(xml_folder):
    """
    Loads extrinsic parameters from the .xml files in the given xml folder.
    """
    # read the data from the files
    print('Loading extrinsic .xml files...')
    rt_file = cv2.FileStorage(os.path.join(xml_folder, 'extrinsics.xml'), cv2.FILE_STORAGE_READ)
    rts = rt_file.getNode('rt_matrix_array').mat()
    rt_file.release()    
    
    # return the parameters
    return rts


def read_extset_measures(extset_folder):
    """
    Read values from the measures.txt file inside the extset folder.
    """
    measures = {}
    with open(os.path.join(extset_folder, 'measures.txt'), 'r') as measures_file:
        for line in measures_file:
            k, v = line.split(', ')
            measures[k] = float(v)
    return measures


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


def run_extrinsic_calibration(mtx, dist, extset_folder, xml_folder):
    """
    Executes the extrinsic calibration procedure using the images stored in images/extset/.
    The procedure will be executed 3 times, with images from each setn/ folder, and the .xml files will be saved to xml/.
    """
    # list of extrinsic matrices
    rts = []
    ds = []
    
    # info from measures file
    measures = read_extset_measures(extset_folder)
    
    # number of chessboard corners
    chess_dim = (8, 6)
    num_points = chess_dim[0]*chess_dim[1]
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # run the calibration procedure 3 times
    print('Starting extrinsic calibration...')
    for i in range(0, 3):
    
        # prepare object points array
        obj_point = np.zeros((num_points, 3), np.float32)
        obj_point[:, :2] = np.mgrid[0:chess_dim[0], 0:chess_dim[1]].T.reshape(-1, 2) * measures['pattern']
        
        # get images from set folder
        set_path = os.path.join(extset_folder, 'set{}'.format(i+1))
        fnames = os.listdir(set_path)
        
        # for all images in the folder...
        for fname in sorted(fnames):
            
            # distance to pattern
            ds.append(measures[fname])
            
            # read image and convert it to grayscale
            img = cv2.imread(os.path.join(set_path, fname))
            img = resize_custom(img, 'qhd', verbose=False)
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
            # axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1, 3) * measures['pattern']
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
            
                
        # print message
        print('Progress: {}/3'.format(i+1))
    
    # calculates mean and standard deviation for the rotation and translation matrix
    rt_mean = [np.mean(rts[0:3], axis=0), np.mean(rts[3:6], axis=0), np.mean(rts[6:9], axis=0)]
    rt_dev = [np.std(rts[0:3], axis=0), np.std(rts[3:6], axis=0), np.std(rts[6:9], axis=0)]
    
    # calculates the mean and standard deviation for the measured distance  
    d_mean = [np.mean(ds[0:3]), np.mean(ds[3:6]), np.mean(ds[6:9])]
    d_dev = [np.std(ds[0:3]), np.std(ds[3:6]), np.std(ds[6:9])]
    
    # calculates the mean and standard deviation for the norm of t
    nts = [np.linalg.norm(rt[:, 3]) for rt in rts]
    nt_mean = [np.mean(nts[0:3]), np.mean(nts[3:6]), np.mean(nts[6:9])]
    nt_dev = [np.std(nts[0:3]), np.std(nts[3:6]), np.std(nts[6:9])]
    
    # print the results
    # print('ds: {}'.format(ds))
    # print('nts: {}'.format(nts))
    # print('d_mean: {}'.format(d_mean))
    # print('d_dev: {}'.format(d_dev))
    # print('nt_mean: {}'.format(nt_mean))
    # print('nt_dev: {}'.format(nt_dev))
    
    # saves the parameters to xml files
    print('Saving .xml files...')
    rt_file = cv2.FileStorage(os.path.join(xml_folder, 'extrinsics.xml'), cv2.FILE_STORAGE_WRITE)
    rt_file.write('rt_matrix_array', np.asarray(rts))
    rt_file.release()