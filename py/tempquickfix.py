import os
f = open('newtargetlist.txt', 'r')
for obj in f:
    objint = int(obj)
    print objint
    print os.path.exists('newnew-{}\n.pickle'.format(objint))
    print '50tau-{}.pickle'.format(objint)
    os.rename('newnew-{}\n.pickle'.format(objint), '50tau-{}.pickle'.format(objint))
