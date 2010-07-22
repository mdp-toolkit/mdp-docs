# Generate "demo.py" using code snippets in "tutorial.rst". To ignore
# a code snippet precede its declaration with the following RST directive:
#.. raw:: html
#
#   <!-- ignore -->
#
#::
#    >>> code()
#
# REMEMBER: The ..raw directive must be INDENTED as the code snippet!

tut = file('tutorial.rst', 'r')
demo = file('demo.py', 'w')
code = False
ignore = False
wrote_lines = 0
read_lines = 0

print 'Reading from "'+tut.name+'", writing to "'+demo.name+'"'
# demo.write('import numpy\n')
for line in tut:
    read_lines += 1
    # strip trailing and preceding blanks
    line = line.strip()
    # look for the "ignore" directive, only if we are not in
    # the middle of an ignored snippet
    if not ignore:
        ignore = line == '<!-- ignore -->'
    # if we are not in the middle of a code snippet,
    # we have to locate the characters '::' at the beginning of a line
    # that's the start of a code snippet, otherwise ignore the line
    if not code:
        code = line.startswith('::')
        if code:
            # we are entering a code snippet, ignore the next line:
            # it's blank!
            dummy = tut.next()
            read_lines += 1
    else:
        if len(line) == 0:
            # that's a blank line: it's the end of the code snippet
            code = False
            ignore = False
        elif ignore:
            # ignore the code snippet
            pass
        elif line.startswith('>>> ') or line.startswith('... '):
            # remove '>>>' or '...', ignore the first blank and write the rest
            out = line.lstrip('>')
            out = out.lstrip('.')
            demo.write(out[1:]+'\n')
            wrote_lines += 1
        else:
            # must be a comment or an interpreter's output.
            # we'll write it as a comment
            demo.write('# '+line+'\n')
            wrote_lines += 1
tut.close()
demo.close()
print 'Lines read =', read_lines
print 'Lines wrote =', wrote_lines
