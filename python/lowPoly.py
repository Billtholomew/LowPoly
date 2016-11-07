import argparse
import time
import sys
import cv2
import numpy as np
from scipy.spatial import Delaunay


def color_triangles(tris, im):
    # 34.75% of this function's time
    triangle_vertices = map(lambda vertices: np.array([vertices]), tris.points[tris.vertices].astype(np.int32))

    # 9.50% of this function's time
    bounding_rectangles = map(lambda vertices: cv2.boundingRect(vertices), triangle_vertices)

    # 33.77% of this function's time
    triangle_colors = map(lambda (c, r, w, h): cv2.mean(im[r:(r + h), c:(c + w), :]), bounding_rectangles)

    im_low_poly = np.zeros(im.shape)

    # 18.69% of this function's time
    map(lambda (vertices, color): cv2.fillConvexPoly(im_low_poly, vertices, color),
        zip(triangle_vertices, triangle_colors))

    return im_low_poly.astype(np.uint8)


# auto canny function
# from http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(im, sigma=0.33):
    v = np.median(im)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(im, lower, upper)


def get_triangulation(im, sigma=0.33):
    edges = auto_canny(im, sigma)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    epsilons = map(lambda x: 0.001 * cv2.arcLength(x, True), contours)
    contours = map(lambda (contour, epsilon): cv2.approxPolyDP(contour, epsilon, True), zip(contours, epsilons))
    pts = np.vstack(map(lambda c: np.array(c).reshape((-1, 2)), contours))
    # now add the four corners, to get better results at corners
    # note that contours is in [x, y] NOT [r, c], so when adding corners, we also need to flip the r/c
    row_max, col_max, channels = im.shape
    corners = np.vstack(([0, 0], [0, row_max], [col_max, 0], [col_max, row_max]))
    pts = np.vstack((pts, corners))

    tris = Delaunay(pts)
    return tris


def process_image(oim):

    t = time.time()
    sigma = 0.50
    triangle_data = get_triangulation(oim, sigma)
    im_low_poly = color_triangles(triangle_data, oim)
    process_time = time.time() - t

    return im_low_poly, process_time


def get_image(file_name=None, camera=None, resize_factor=1):
    if camera is not None:
        retval, oim = camera.read()
    elif file_name is not None:
        oim = cv2.imread(file_name, flags=cv2.CV_LOAD_IMAGE_COLOR)  # ensure we read it in as a color image
    else:
        raise Exception('No source provided, cannot read in image')
    # scale image down for faster processing
    if resize_factor != 1:
        oim = cv2.resize(oim, (0, 0), fx=resize_factor, fy=resize_factor,
                         interpolation=cv2.INTER_LINEAR)
    return oim


def process_from_camera_feed(resize_factor=1):
    camera = None
    try:
        camera_port = 0
        camera = cv2.VideoCapture(camera_port)
        start_time = time.time()
        time.sleep(1)
        total_frames = 0
        while True:
            total_frames += 1
            frame_rate = total_frames / (time.time() - start_time)
            if frame_rate < 25:
                continue
            oim = get_image(camera=camera, resize_factor=resize_factor)
            im_low_poly, process_time = process_image(oim)

            # scale image back up
            if resize_factor != 1:
                oim = cv2.resize(oim, (0, 0), fx=resize_factor, fy=resize_factor,
                                 interpolation=cv2.INTER_CUBIC)
                im_low_poly = cv2.resize(im_low_poly, (0, 0), fx=resize_factor, fy=resize_factor,
                                         interpolation=cv2.INTER_LINEAR)

            cv2.namedWindow('Compare', flags=cv2.WINDOW_NORMAL)
            compare = np.hstack([oim, im_low_poly])
            cv2.imshow('Compare', compare)
            if cv2.waitKey(30) >= 0:
                break
    except Exception, e:
        print e
    finally:
        del camera
        cv2.destroyAllWindows()


def process_from_single_file(in_name=None, out_name=None, show=False, resize_factor=1):
    oim = get_image(file_name=in_name, resize_factor=resize_factor)
    im_low_poly, _ = process_image(oim)

    # scale image back up
    if resize_factor != 1:
        oim = cv2.resize(oim, (0, 0), fx=1 / resize_factor, fy=1 / resize_factor,
                         interpolation=cv2.INTER_CUBIC)
        im_low_poly = cv2.resize(im_low_poly, (0, 0), fx=1 / resize_factor, fy=1 / resize_factor,
                                 interpolation=cv2.INTER_NEAREST)

    if show:
        cv2.namedWindow('Compare', flags=cv2.WINDOW_NORMAL)
        compare = np.hstack([oim, im_low_poly])
        cv2.imshow('Compare', compare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if out_name is not None:
        cv2.imwrite(out_name, im_low_poly)


def main(in_name=None, out_name=None, show=False, resize_factor=1):
    try:
        if in_name is not None:
            process_from_single_file(in_name=in_name, out_name=out_name, show=show, resize_factor=resize_factor)
        else:
            process_from_camera_feed(resize_factor=1)
    except Exception, e:
        print e
    finally:
        cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Artistically creates a low-polygon version of an image')
parser.add_argument('--source', '-s', nargs='?', choices=['camera', 'file'], dest='source', required=True,
                    help='where to get image from, if "file" --input <full path> is required')
parser.add_argument('--input', '-i', nargs='?', dest='iName', const=str, default=None,
                    help='full path of image to process')
parser.add_argument('--output', '-o', nargs='?', dest='oName', const=str, default=None,
                    help='full path of image to process')
parser.add_argument('--view', '-v', nargs='?', dest='show', const=bool, default=True,
                    help='boolean to view image or not, Default: True')
args = parser.parse_args()

if __name__ == '__main__':
    if args.source == 'camera':
        if args.iName is not None:
            print 'Reading from camera. Option "--input/-i', args.fName+'"','will be ignored'
        main(in_name=None, out_name=None, show=True, resize_factor=1)
    elif args.source == 'file':
        if args.iName is not None:
            main(in_name=args.iName, out_name=args.oName, show=args.show, resize_factor=1)
        else:
            print 'ERROR: With source set to "file", --input/-i must be set to the full path to file'