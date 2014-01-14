import os
from grid import make_grid_plots

count = 0

f = open('ids.txt','r')

for line in f:
    objid = int(line)
    make_grid_plots(objid)
    assert(False)
    #UGLY but i'm lazy rn
    try:
        os.mkdir(os.path.expanduser('~/public_html/%d' % objid))
    except:
        pass
    for band in ['u','g','r','i','z']:
        os.rename('grid-%d-%s.png' %(objid,band),os.path.expanduser('~/public_html/%d/%s.png' % (objid,band)))
        os.rename('grid-%d-%s-overlay.png' %(objid,band),os.path.expanduser('~/public_html/%d/%s-overlay.png' % (objid,band)))
    count+=1
    if count==1:
        break

