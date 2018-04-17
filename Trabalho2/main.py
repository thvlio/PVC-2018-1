# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# import configurations file
import config

# import functions
import line
import intrinsic
import extrinsic
import ruler


def check_args(args):
    """
    Process the given arguments and check for errors. Returns mode and smode.
    """
    # check for mode errors
    mode = None
    modes_list = [args['line'], args['intrinsic'], args['extrinsic'], args['ruler']]
    if True not in modes_list:
        print('No operating mode was selected. Use --help to check usage of the program.')
    elif modes_list.count(True) > 1:
        print('Only one operating mode can be selected. Use --help to check usage of the program.')
    else:
        mode = ['line', 'intrinsic', 'extrinsic', 'ruler'][modes_list.index(True)]
        print('Selected ' + mode + ' mode.')
        
    # check for existence of images for calibration and xml files for loading
    if mode == 'intrinsic' or mode == 'extrinsic':
        if args['imageset'] is None:
            print('The imageset folder needs to be provided. Use --help to check usage of the program.')
            mode = None
        else:
            if not os.path.isdir(args['imageset']):
                print('Folder {} not found.'.format(args['imageset']))
                mode = None
            else:
                print('Imageset folder: {}'.format(args['imageset']))
        if args['xml'] is None:
            print('The xml folder needs to be provided. Use --help to check usage of the program.')
            mode = None
        else:
            if not os.path.isdir(args['xml']):
                print('Folder {} not found.'.format(args['xml']))
                mode = None
            else:
                print('Xml folder: {}'.format(args['xml']))
    elif mode == 'ruler':
        if args['imageset'] is None:
            print('The distance name needs to be provided via the imageset argument (dmin_n, dmed_n or dmax_n, where n can be 1, 2 or 3). Use --help to check usage of the program.')
            mode = None
        else:
            if args['imageset'] not in ['dmin_1', 'dmin_2', 'dmin_3', 'dmed_1', 'dmed_2', 'dmed_3', 'dmax_1', 'dmax_2', 'dmax_3']:
                print('The distance name can only be one of the following: dmin_n, dmed_n or dmax_n, where n can only be 1, 2 or 3. Use --help to check usage of the program.')
                mode = None
        if args['xml'] is None:
            print('The xml folder needs to be provided. Use --help to check usage of the program.')
            mode = None
        else:
            if not os.path.isdir(args['xml']):
                print('Folder {} not found.'.format(args['xml']))
                mode = None
            else:
                print('Xml folder: {}'.format(args['xml']))
    elif mode == 'line':
        if args['imageset'] is not None:
            print('The imageset folder does not need to be provided. Ignoring argument. Use --help to check usage of the program.')
        if args['xml'] is not None:
            print('The xml folder does not need to be provided. Ignoring argument. Use --help to check usage of the program.')
              
    # check for streaming mode errors
    smode = None
    smodes_list = [False if args['image'] is None else True,
        False if args['video'] is None else True,
        False if args['webcam'] is None else True]
    if mode == 'line' or mode == 'intrinsic':
        if True not in smodes_list:
            print('No streaming mode was selected. Use --help to check usage of the program.')
        elif smodes_list.count(True) > 1:
            print('Only one streaming mode can be selected. Use --help to check usage of the program.')
        else:
            smode = ['image', 'video', 'webcam'][smodes_list.index(True)]
            print('Selected ' + smode + ' mode.')
            if (smode == 'image' or smode == 'video'):
                print('File to be used: {}'.format(args[smode]))
    elif mode == 'extrinsic':
        if smodes_list.count(True) > 0:
            print('No streaming mode needs to be selected when operating in extrinsic mode. Ignoring argument. Use --help to check usage of the program.')
    elif mode == 'ruler':
        if True not in smodes_list:
            print('No streaming mode was selected. For ruler mode, only images can be used. Use --help to check usage of the program.')
        elif smodes_list.count(True) > 1:
            print('Only one streaming mode can be selected. For ruler mode, it must be image mode. Use --help to check usage of the program.')
        else:
            smode = ['image', 'video', 'webcam'][smodes_list.index(True)]
            if smode != 'image':
                print('For ruler mode, only image mode can be used. Use --help to check usage of the program.')
                smode = None
            else:
                print('Selected ' + smode + ' mode.')
                print('File to be used: {}'.format(args[smode]))
        
    # check for file existence and webcam device
    if (smode == 'image' or smode == 'video') and not os.path.isfile(args[smode]):
        print('File {} not found.'.format(args[smode]))
        smode = None
    if smode == 'webcam':
        print('Camera device selected: {}'.format(args[smode]))

    # return mode and smode
    return mode, smode


def main():
    """
    Main function.
    """
    # setup argument parser
    ap = argparse.ArgumentParser('Select a mode of operation and streaming mode.')
    
    # modes of operation
    ap.add_argument('--line', action='store_true',
        help='Line mode, in which pixel distance is measured. Can be used with image, video or webcam.')
    ap.add_argument('--intrinsic', action='store_true',
        help='Intrinsic parameters calibration mode, in which intrinsic calibration parameters are calculated and distance can be measure on two windows. Path to xml folder and path to intset folder must be specified. Can be used with image, video or webcam.')
    ap.add_argument('--extrinsic', action='store_true',
        help='Extrinsic parameters calibration mode, in which extrinsic calibration parameters are calculated. Path to xml folder and path to extset folder must be specified. Does not need a streaming mode.')
    ap.add_argument('--ruler', action='store_true',
        help='Ruler mode, in which distance in the world frame is measured based on pixel distance. Xml folder needs to be specified. Distance name must be specified with imagset parameter. Can only be used with image.')
    
    # paths to set folders and xml folders
    ap.add_argument('--imageset', type=str,
        help='Path to imageset folder, in which there are n set folders containing the calibration images.')
    ap.add_argument('--xml', type=str,
        help='Path to xml folder, in which the calibration parameters are stored.')
    
    # streaming modes
    ap.add_argument('--image', type=str,
        help='Run program using image. Image path must follow')
    ap.add_argument('--video', type=str,
        help='Run program using video. Video path must follow')
    ap.add_argument('--webcam', type=int,
        help='Run program using webcam. Device number must follow')
    
    # parse the arguments
    args = vars(ap.parse_args())
    
    # check the arguments for errors
    mode, smode = check_args(args)

    # run selected mode
    if mode is not None:
        
        if mode == 'line' and smode is not None:
            line.display(mode=smode, fpath=args[smode])
                
        elif mode == 'intrinsic' and smode is not None:
            mtx, dist = intrinsic.run_intrinsic_calibration(intset_folder=args['imageset'], xml_folder=args['xml'])
            intrinsic.display(mode=smode, fpath=args[smode], mtx=mtx, dist=dist)
                    
        elif mode == 'extrinsic' and smode is None:
            mtx, dist = intrinsic.load_intrinsic_parameters(xml_folder=args['xml'])
            extrinsic.run_extrinsic_calibration(mtx=mtx, dist=dist, extset_folder=args['imageset'], xml_folder=args['xml'])
                        
        elif mode == 'ruler' and smode == 'image':
            mtx, dist = intrinsic.load_intrinsic_parameters(xml_folder=args['xml'])
            rts = extrinsic.load_extrinsic_parameters(xml_folder=args['xml'])
            rt = rts[['dmin_1', 'dmin_2', 'dmin_3', 'dmed_1', 'dmed_2', 'dmed_3', 'dmax_1', 'dmax_2', 'dmax_3'].index(args['imageset'])]
            ruler.display(mode=smode, fpath=args[smode], mtx=mtx, dist=dist, rt=rt)
            
if __name__ == '__main__':
    main()