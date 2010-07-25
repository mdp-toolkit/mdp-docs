import sys

def do_not_skip(app, what, name, obj, skip, options):
    ret = skip
    if name.startswith('_') and not name.startswith('__'):
        ret = False
    #print "autodoc-skip-member", name, skip, '->', ret
    return ret

def setup(app):
    app.connect('autodoc-skip-member', do_not_skip)
    return
