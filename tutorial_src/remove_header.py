import os
import sys

# input filename
inp_fl = file(sys.argv[1], 'r')
inp = inp_fl.read()
inp_fl.close()

# tutorial filename
tut = sys.argv[2]

# html_src directory
html_src = '../html_src/'

# read head and foot
head_fl = file(html_src+'header_and_side.html')
head = head_fl.read()
head_fl.close()
foot_fl = file(html_src+'footer.html')
foot = foot_fl.read()
foot_fl.close()

# find body
start_idx = inp.find('<body>')
stop_idx = inp.find('</body>')

# enclose it with our header and footer
content = inp[:start_idx+6] + head + inp[start_idx+6:stop_idx] + foot

out = file('../html/'+tut, 'w')
out.write(content)
out.close()
