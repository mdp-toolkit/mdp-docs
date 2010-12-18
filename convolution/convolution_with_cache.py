import mdp
import numpy
import pylab

CACHEDIR = '/tmp/mdpcache'

# load image
im = pylab.imread('lena.png')
# transform to gray
im = numpy.sqrt((im[:,:,:3]**2.).mean(2))

# create Gabor filters bank
pi = numpy.pi
orientations = [0., pi/4., pi/2., pi*3./4.]
freq = 1./10
phi = pi/2.
size = (20, 20)
sgm = (5., 3.)

nfilters = len(orientations)
gabors = numpy.empty((nfilters, size[0], size[1]))
for i, alpha in enumerate(orientations):
    gabors[i,:,:] = mdp.utils.gabor(size, alpha, phi, freq, sgm)

# we fake having a large number of images by multiplying one image
# we've got and adding noise
x = mdp.utils.lrep(im, 3)

# convolve image with gabors
node = mdp.nodes.Convolution2DNode(gabors, mode='valid', boundary='fill',
                                   fillvalue=0, output_2d=False)


# ----- from here we demonstrate the caching mechanism
from timeit import Timer
timer = Timer("node.execute(x)", "from __main__ import node, x")
print '- first uncached execution'
print '  ', timer.repeat(1, 1), 'sec'


with mdp.caching.cache(cachedir=CACHEDIR,
                       cache_classes=[mdp.nodes.Convolution2DNode]):
    print '- caching mechanism activated'
    print "- second execution, uncached if it's the first time the script is run"
    print '  ', timer.repeat(1, 1), 'sec'
    print '- third execution, this time cached'
    print '  ', timer.repeat(1, 1), 'sec'
