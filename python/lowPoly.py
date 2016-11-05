import time
import sys
import cv2
import numpy as np
from scipy.spatial import Delaunay


def color_triangles(tris, im):
    # 0.123 s
    triangle_vertices = map(lambda vertices: np.array([vertices]), tris.points[tris.vertices].astype(np.int32))

    # 0.035 s
    bounding_rectangles = map(lambda vertices: cv2.boundingRect(vertices), triangle_vertices)

    # 0.123 s
    triangle_colors = map(lambda (c, r, w, h): cv2.mean(im[r:(r + h), c:(c + w), :]), bounding_rectangles)

    im_low_poly = np.zeros(im.shape)

    # 0.067 s
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

    tt = time.time()
    tris = Delaunay(pts)
    print time.time() - tt
    return tris


def wrapper(in_name, out_name=None, show=False):

    oim = cv2.imread(in_name, flags=cv2.CV_LOAD_IMAGE_COLOR)  # ensure we read it in as a color image

    t = time.time()
    sigma = 0.33
    triangle_data = get_triangulation(oim, sigma)
    im_low_poly = color_triangles(triangle_data, oim)
    print 'Total:', time.time() - t


    if show:
        cv2.namedWindow('Compare', flags=cv2.WINDOW_NORMAL)
        compare = np.hstack([oim, im_low_poly])
        cv2.imshow('Compare', compare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if out_name is not None:
        cv2.imwrite(out_name, im_low_poly)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage'
        print 'lowPoly.py <infile> [outfile]'
        print 'Example'
        print 'lowPoly.py ../data/Lenna.png'
        print 'lowPoly.py ../data/Lenna.png imOut.png'
    else:
        inFile = sys.argv[1]
        outFile = None
        if len(sys.argv) == 3:
            outFile = sys.argv[2]
        wrapper(in_name=inFile, out_name=outFile, show=True)
