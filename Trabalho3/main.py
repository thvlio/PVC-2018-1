# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# import configurations file
import config

# import functions
import depth


def main():
    """
    Main function.
    """
    # setup argument parser
    ap = argparse.ArgumentParser('Select a mode of operation and streaming mode.')
    
    # mode arguments
    ap.add_argument('--gen-maps', type=str, nargs=2,
        help='Generate disparity and depth maps. Expects two image paths (first L then R).')
    ap.add_argument('--est-params', action='store_true',
        help='If present, estimate K and rectify the images before computing the maps.')
    ap.add_argument('--load-maps', type=str, nargs=2,
        help='Execute ruler mode, in which an object can be measured.')
    
    # optional arguments for the modes
    ap.add_argument('--show-maps', action='store_true',
        help='If present, show the disparity and depth maps when computing them.')
    ap.add_argument('--desc', type=str,
        help='Descriptor. To use ORB with 1000 max points, use orb-1000.')
    ap.add_argument('--show-matches', action='store_true',
        help='If present, shows the images containing the matches.')
    ap.add_argument('--write-maps', action='store_true',
        help='If present, writes the disparity and depth maps to disk.')
    ap.add_argument('--scale', type=float, default=0.2,
        help='Defines the scale of the maps (smaller is faster).')
    ap.add_argument('--window-size', type=float, default=64,
        help='Window size for template matching.')
    ap.add_argument('--min-disp', type=int, default=0,
        help='Sets the minimum disparity value.')
    ap.add_argument('--max-disp', type=int, default=np.inf,
        help='Sets the maximum disparity value.')
    ap.add_argument('--tol', type=float, default=5.0,
        help='Tolerance for outlier filtering after calculating disparities.')
    ap.add_argument('--use-sgbm', action='store_true',
        help='Use StereoSGBM instead of the hand coded matcher.')
    
    # parse the arguments
    args = vars(ap.parse_args())

    # run req 1 or 2
    if args['gen_maps'] is not None:
        
        # reads the images
        imgL = cv2.imread(args['gen_maps'][0])
        imgR = cv2.imread(args['gen_maps'][1])
        
        # dont estimate the parameters (req 1)
        if not args['est_params']:
            imgL_rec = imgL.copy()
            imgR_rec = imgR.copy()
            f = 15
            b = 120
            
        # estimate the parameters (req 2)
        else:
            imgL_c = imgL.copy()
            imgR_c = imgR.copy()
            F, ptsL, ptsR = depth.find_correspondences(imgL_c, imgR_c, args['desc'], args['show_matches'])
            imgL_rec, imgR_rec, f, b, _, _ = depth.compute_stereo_params(imgL_c, imgR_c, F, ptsL, ptsR)
            
        # use stereosgbm from opencv
        if args['use_sgbm']:
            
            # create and configure the the stereo matcher
            stereo = cv2.StereoSGBM_create()
            stereo.setMinDisparity(1)
            stereo.setNumDisparities(128)
            stereo.setBlockSize(21)
            stereo.setSpeckleRange(2)
            stereo.setSpeckleWindowSize(51)

            # compute the disparity map
            grayL = cv2.cvtColor(imgL_rec, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR_rec, cv2.COLOR_BGR2GRAY)
            disp_map = stereo.compute(grayL, grayR) / 2048
        
        # otherwise, use the hand coded matcher
        else:
            disp_map = None
        
        # get the disparity and depth maps
        disp_map, disp_map_normed, depth_map, depth_map_normed, Zmin, Zmax = depth.compute_disparity_and_depth(
            imgL_rec, imgR_rec, scale=args['scale'], window_size=args['window_size'],
            min_disp=args['min_disp'], max_disp=args['max_disp'], tol=args['tol'],
            f=f, b=b, show_maps=args['show_maps'], d_map=disp_map)
            
        # save the maps to disk
        if args['write_maps']:
            img_name = args['gen_maps'][0].split(os.path.sep)
            raw_name = img_name[-1][:-5]
            suffix = ''
            if args['use_sgbm']:
                suffix = '_sgbm'
            cv2.imwrite('imgs/{}_disp{}.jpg'.format(raw_name, suffix), disp_map_normed)
            cv2.imwrite('imgs/{}_depth{}.jpg'.format(raw_name, suffix), depth_map_normed)
    
    # run req 3    
    elif args['load_maps'] is not None:
        
        # reads both images
        imgL = cv2.imread(args['load_maps'][0])
        imgR = cv2.imread(args['load_maps'][1])
        
        # estimate params
        imgL_c = imgL.copy()
        imgR_c = imgR.copy()
        F, ptsL, ptsR = depth.find_correspondences(imgL_c, imgR_c, args['desc'], args['show_matches'])
        _, _, f, b, H1, H2 = depth.compute_stereo_params(imgL_c, imgR_c, F, ptsL, ptsR)
        
        # display
        depth.display(imgL, imgR, H1, H2, f, b)
    
if __name__ == '__main__':
    main()