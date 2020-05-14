import mdp
import numpy
import pylab

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

# convolve image with gabors
node = mdp.nodes.Convolution2DNode(gabors, mode='valid', boundary='fill',
                                   fillvalue=0, output_2d=False)
cim = node.execute(im[numpy.newaxis,:,:])

# show convolved images
def grayshow(im):
    pylab.gray()
    pylab.imshow(im, hold=False)
    pylab.axis('off')
    pylab.draw()

pylab.figure(1)
grayshow(im)

pylab.figure(2)
pylab.clf()
for i in range(nfilters):
    pylab.subplot(2,2,i+1)
    grayshow(cim[0,i,:])
