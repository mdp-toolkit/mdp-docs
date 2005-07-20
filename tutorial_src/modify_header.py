import os

# where description tags are
descr_dir = '../html_src/descriptions/'
fl = file('tutorial.txt', 'r')
content = fl.read()
fl.close()

#get descriptions
execfile(descr_dir+'tutorial_head.py')

headings = """
.. meta::
   :description: %s
   :keywords: %s
 
"""%(Description, Keywords)

print headings+content

# should add
#
#"""
#  .. title::
#     %s
#
#""" %(Title)
