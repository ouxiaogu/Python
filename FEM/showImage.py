import os, os.path
import matplotlib.pyplot as plt
import sysimp

imagepath=sys.argv[1]
#imfile = os.path.join(workpath, 'Lenna.png')
img = plt.imread(imagepath)
plt.imshow(img)
plt.show()