import math
import time
import cv2
import numpy as np
import scipy
from scipy import ndimage
from scipy.spatial import Delaunay
import threading
import Queue
import sys

# create a new image using the average color for each triangle
def getLowPoly(tris,hipo):
  # define this in the other function so we can use the same vars
  def colorImage(tris,rv=None):
    for tri in tris:
      lopo[tridex==tri,:] = np.mean(hipo[tridex==tri,:],axis=0)
  # subs is the subscripts of all parts of the image
  # so tridex holds, for each pixel in [r*row+c] organization, each triangle
  # for each pixel
  # it's kind of a lookup table
  subs = np.transpose(np.where(np.ones(hipo.shape[:2])))
  subs = subs[:,:2]
  tridex = tris.find_simplex(subs)
  tridex = tridex.reshape(hipo.shape[:2])
  pTris = np.unique(tridex)
  lopo = np.zeros(hipo.shape)

  colorImage(pTris)
  lopo = lopo.astype(np.uint8)
  return lopo

def getTriangulation(im,a=50,b=55,c=0.15,debug=False):
  edges = cv2.Canny(im,a,b)
  nTris = c
  #nPts = np.ceil(np.sqrt(24*nTris)*3.33)
  nPts = np.where(edges)[0].size*c
  r,c = np.nonzero(edges)
  rnd = np.zeros(r.shape)==1
  rnd[:nPts] = True
  np.random.shuffle(rnd)
  r = r[rnd]
  c = c[rnd]
  sz = im.shape
  rMax = sz[0]
  cMax = sz[1]
  pts = np.vstack([r,c]).T
  pts = np.vstack([pts,[0,0]])
  pts = np.vstack([pts,[0,cMax]])
  pts = np.vstack([pts,[rMax,0]])
  pts = np.vstack([pts,[rMax,cMax]])
  tris = Delaunay(pts)
  if debug:
    cv2.imshow('e',edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()        
    im2 = np.zeros(sz)
    im2[r,c] = 255
    cv2.imshow('tests',im2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  return tris

def preProcess(hipo,newSize=None):
  # handle gray scale
  if hipo.shape[2]==1:
    hipo = hipo.dstack([hipo,hipo,hipo],axis=2)
  if newSize is not None:
    scale = newSize/float(np.max(hipo.shape[:2]))
    hipo = cv2.resize(hipo,(0,0),fx=scale,fy=scale,interpolation = cv2.INTER_CUBIC)
  return hipo

def wrapper(inName,a,b,c,outName=None,show=False):
  
  hipo = cv2.imread(inName)

  hipo = preProcess(hipo,newSize=500)

  t = time.time()
  tris = getTriangulation(hipo,a,b,debug=False)
  print time.time()-t
  t = time.time()
  lopo = getLowPoly(tris,hipo)
  print time.time()-t

  if show:
    compare = np.hstack([hipo,lopo])
    cv2.imshow('Compare',compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  if outName is not None:
    cv2.imwrite(outName,lopo)

if __name__ == "__main__":
  if len(sys.argv)<2:
    print 'Usage'
    print 'lowPoly.py <infile> [outfile]'
    print 'Example'
    print 'lowPoly.py ../data/Lenna.jpg'
    print 'lowPoly.py ../data/Lenna.jpg imOut.png'
  else:
    inFile = sys.argv[1]
    outFile = None
    if len(sys.argv)==3:
      outFile = sys.argv[2]
    wrapper(inName=inFile,a=100,b=75,c=0.05,outName=outFile,show=True)
