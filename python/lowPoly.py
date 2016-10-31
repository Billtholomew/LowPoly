import time
import sys
import cv2
import numpy as np
from scipy.spatial import Delaunay


# create a new image using the average color for each triangle
def getLowPoly(tris, im):

    # both triangle_vertices and bounding rectangles "share" the first element
    # that is, both refer to the triangle index, for triangle_vertices this is the triangle that it is,
    # for bounding_rectangles this is the triangle that the rectangle bounds

    triangle_vertices = {triangle_id: np.array([tris.points[vertices]]).astype(np.int32)
                         for triangle_id, vertices in enumerate(tris.vertices)}

    bounding_rectangles = {triangle_id: cv2.boundingRect(vertices)
                           for triangle_id, vertices in triangle_vertices.iteritems()}

    triangle_means = {triangle_id: cv2.mean(im[r:(r + h), c:(c + w), :])
                      for triangle_id, (c, r, w, h) in bounding_rectangles.iteritems()}

    im_low_poly = np.zeros(im.shape)
    map(lambda (triangle_id, color_mu):
        cv2.fillConvexPoly(im_low_poly, triangle_vertices[triangle_id], color_mu),
        triangle_means.iteritems())

    return im_low_poly.astype(np.uint8)


def im_close(im, n=5):
    im = cv2.dilate(im, np.ones((n, n), np.uint8), iterations=1)
    im = cv2.erode(im, np.ones((n, n), np.uint8), iterations=1)
    return im


def im_open(im, n=5):
    im = cv2.erode(im, np.ones((n, n), np.uint8), iterations=1)
    im = cv2.dilate(im, np.ones((n, n), np.uint8), iterations=1)
    return im


def threshold_image(im, color_mu, color_std=0, sigma=1):
    nim = im.copy()
    _, ima = cv2.threshold(im, color_mu - (color_std * sigma), 255, cv2.THRESH_BINARY)
    _, imb = cv2.threshold(im, color_mu + (color_std * sigma), 255, cv2.THRESH_BINARY_INV)
    nim[np.equal(ima, imb)] = 255
    nim[np.not_equal(ima, imb)] = 0
    nim = im_open(nim)
    nim = im_open(nim)
    return nim


def im_mask(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    card_color_mu, card_color_std = cv2.meanStdDev(im_gray)
    return threshold_image(im_gray, card_color_mu, card_color_std).astype(np.uint8)


# auto canny function
# from http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(im, sigma=0.33):
    v = np.median(im)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(im, lower, upper)


def get_triangulation(im, debug=False):
    if debug:
        def nothing(x):
            pass
        cv2.namedWindow("c")
        cv2.createTrackbar("Epsilon Factor", "c", 0, 100, nothing)
        cv2.createTrackbar("Edge Sigma", "c", 0, 100, nothing)

    edges = auto_canny(im)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if 2 < cv2.contourArea(contour)]
    epsilons = map(lambda x: 1 * cv2.arcLength(x, True) / 100.0, contours)
    contours = map(lambda (contour, epsilon): cv2.approxPolyDP(contour, epsilon, True), zip(contours, epsilons))
    pts = np.vstack(map(lambda c: np.array(c).reshape((-1, 2)), contours))
    # now add the four corners, to get better results at corners
    sz = im.shape
    rMax = sz[0]
    cMax = sz[1]
    pts = np.vstack([pts, [0, 0]])
    pts = np.vstack([pts, [0, rMax]])
    pts = np.vstack([pts, [cMax, 0]])
    pts = np.vstack([pts, [cMax, rMax]])

    while debug:
        e_factor = cv2.getTrackbarPos("Epsilon Factor", "c") / 100.0
        sigma = cv2.getTrackbarPos("Edge Sigma", "c") / 100.0
        edges = auto_canny(im, sigma)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if 2 < cv2.contourArea(contour)]
        epsilons = map(lambda x: e_factor * cv2.arcLength(x, True), contours)
        contours = map(lambda (contour, epsilon): cv2.approxPolyDP(contour, epsilon, True), zip(contours, epsilons))
        pts = np.vstack(map(lambda c: np.array(c).reshape((-1, 2)), contours))
        contour_image = np.zeros(im.shape)
        cv2.drawContours(contour_image, contours, -1, (255, 0, 0))
        cv2.imshow('c', contour_image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    tris = Delaunay(pts)

    return tris


def preProcess(hipo, newSize=None):
    # handle gray scale
    if hipo.shape[2] == 1:
        hipo = hipo.dstack([hipo, hipo, hipo], axis=2)
    if newSize is not None:
        scale = newSize / float(np.max(hipo.shape[:2]))
        hipo = cv2.resize(hipo, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return hipo


def wrapper(inName, outName=None, show=False):
    hipo = cv2.imread(inName)

    hipo = preProcess(hipo, newSize=500)

    t = time.time()
    tris = get_triangulation(hipo, debug=False)
    lopo = getLowPoly(tris, hipo)
    print 'Time:',time.time() - t
    if show:
        compare = np.hstack([hipo, lopo])
        cv2.imshow('Compare', compare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if outName is not None:
        cv2.imwrite(outName, lopo)


if __name__ == "__main__":
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
        wrapper(inName=inFile, outName=outFile, show=True)
