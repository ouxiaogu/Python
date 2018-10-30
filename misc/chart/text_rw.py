import numpy
import pylab

# Open file
f  =  open("C:/Localdata/D/Note/Python/chart/table1.txt",'r')

# Read and ignore header lines
header1 = f.readline()

# Loop over lines and extract variables of interest
gauges = []
for line in f:
    print(repr(line)) # repr : object representations as a whole string
    line = line.strip()
    cols = line.split('\t')
    source = {}
    source['gauge'] = cols[0]
    print(source['gauge'])
    source['draw_cd'] = float(cols[1])
    source['range_min'] = float(cols[2])
    source['range_max'] = float(cols[3])
    source['wafer_cd'] = float(cols[4])
    source['model_cd'] = float(cols[5])
    source['model_err'] = float(cols[6])
    gauges.append(source)
    
#print(type(gauges))
#print(gauges[0])

# iterate for list & dict
for value in gauges:
    #print(value)
    #print(type(value))
    for key1, val1 in value.items():
        print(key1, val1)
    
    

