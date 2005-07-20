import sys

fid = file(sys.argv[1] ,'r')
code = 0
for line in fid:
        if line.strip() == '':
                print line,
        elif line[0] != '#':
                if not code:
                        print '::\n\n',
                        code = 1
                print '    >>> '+line,
        else:
                if code:
                        print
                        code = 0
                str = line[1:].replace('#', '').lstrip()
                if str=='':
                        print
                else:
                        print str,
fid.close()
