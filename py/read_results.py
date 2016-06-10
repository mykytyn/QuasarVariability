import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import triangle

name = 'deltaln.txt'
tau = []
lnprob = []
meanu = []
meang = []
meanr = []
meani = []
meanz = []
ar = []
alpha = []
tau = []
quasars = []
f = open(name)
line = f.next()[1:-2]
print line
temp = line.split(',')
quasar = [float(temp[2]),float(temp[3]),float(temp[4]),float(temp[5]),float(temp[6]),float(temp[7]),float(temp[8]),float(temp[9])]

#quasars.append(quasar)
#stack = np.column_stack(quasars).T
#print stack[:,8]
#print stack.shape
#figure = triangle.corner(stack,labels=['Lnprob', 'Mean u', 'Mean g', 'Mean r', 'Mean i', 'Mean z', '$A_r$', '$\\alpha$','$\\tau$'])
#figure.savefig('test-triangle2.png')

