# generate an array containing a black text on white background centered
# with a half font size margin.

fontpath = "Georgia_Bold.ttf" # which truetype font to use
fontsize = 300
text = "MDP"

import Image, ImageDraw, ImageFont, pickle, numpy, pylab
print 'Generate image.'
# size of the starting image is enough to contain the text
imgsize = (len(text)*fontsize, len(text.splitlines())*fontsize)
# mode 'L' is 8-bit pixels (signed integers), black & white
image = Image.new("L", imgsize)
draw = ImageDraw.Draw(image)
draw.text((0,0),text,font=ImageFont.truetype(fontpath,fontsize),fill=1)
# crop the image to its natural bounding box
image = image.crop(image.getbbox())
# flip the image top-down (the origin for arrays is in the lower-left
# corner)
image = image.transpose(Image.FLIP_TOP_BOTTOM)
# read image as an array of ints
x = numpy.fromstring(image.tostring(), numpy.int8)
x.shape = (image.size[1], image.size[0])
# add half the font size as a margin and convert to floats
y = numpy.zeros((x.shape[0]+fontsize, x.shape[1]+fontsize),
                dtype=numpy.int8)
# place the image in the middle of the new array
fs2 = int(fontsize/2)
y[fs2:fs2+x.shape[0], fs2:fs2+x.shape[1]] = x

# now pickle the array for future use
fl = file('text.raw', 'wb')
pickle.dump(y, fl)
fl.close()

# have a look at it
pylab.ioff()
pylab.imshow(y, cmap=pylab.cm.Greys, origin='lower')
pylab.axis('scaled')
pylab.show()

