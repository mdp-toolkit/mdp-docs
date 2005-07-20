import os

# where html sources are
source_dir = 'pages'
# where description tags are
descr_dir = 'descriptions'
# where to put results
out_dir = '../html'

# header and sidebar template
h_side = 'header_and_side.html'
# header of the html document
head = 'head.html'
# footer template
foot = 'footer.html'

# get header and sidebar template contents:
h_side_fl = file(h_side, 'r')
h_side = h_side_fl.read()
h_side_fl.close()

# get footer template contents:
foot_fl = file(foot, 'r')
foot = foot_fl.read()
foot_fl.close()

# get list of all htmls
all_files = os.listdir(source_dir)
sources = [x for x in all_files if x[-5:] == '.html']

for src in sources:
    print 'Processing ', src
    # output string
    out = ''
    tag = src[:-5]
    # get description tags
    execfile(os.path.join(descr_dir, tag+'_head.py'))
    # adjust header of the html document using description tags
    head_fl = file(head, 'r')
    for line in head_fl:
        if line == '<TITLE></TITLE>\n':
            out = out + '<TITLE>' + Title + '</TITLE>\n'
        elif line == '<META NAME="Keywords" CONTENT="">\n':
            out = out + '<META NAME="Keywords" CONTENT="' + Keywords + '">\n' 
        elif line == '<META NAME="Description" CONTENT="">\n':
            out = out + '<META NAME="Description" CONTENT="' + Description + '">\n'
        elif line == '<LINK REL=STYLESHEET TYPE="text/css" HREF="" TITLE="Main Styles">\n':
            out = out + '<LINK REL=STYLESHEET TYPE="text/css" HREF="' + Stylesheet + '" TITLE="Main Styles">\n'
        else:
            out = out+line        
    head_fl.close()
    # put in header and sidebar
    out = out + h_side
    # read in actual content of the page
    content = file(os.path.join(source_dir, src), 'r')
    out = out + content.read()
    content.close()
    # put in footer
    out = out + foot
    # save result
    result = file(os.path.join(out_dir, src), 'w')
    result.write(out)
    result.close
