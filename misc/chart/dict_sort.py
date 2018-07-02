import operator
gauges1 = {'model_cd': ['46.95', '55.66', '65.11', '74.31', '107.78', '138.17', '165.58', '192.99', '327.99', '47.04', '47.92', '52.78', '53.4', '49.05', '51.33', '48.18', '47.24'],
'range_min': ['-5', '-5', '-5', '-5', '-5', '-5.86', '-5.94', '-5.7', '-5.56', '-5', '-5', '-2', '-5', '-2', '-2', '-2', '-2'], 
'range_max': ['5', '5', '5', '5', '5', '5.86', '5.94', '5.7', '5.56', '5', '5', '2', '5', '2', '2', '2', '2'],
'model_error': ['-0.72', '1.33', '0.5', '1.42', '2.94', '0.27', '0.22', '-0.02', '3.05', '-0.16', '1.58', '-0.87', '1.99', '-1.16', '1.87', '1.73', '1.99'],
'wafer_cd': ['0', '54.33', '64.61', '0', '104.84', '137.9', '165.36', '193.01', '324.94', '47.2', '46.34', '53.65', '51.41', '50.21', '49.46', '46.45', '45.25'], 
'gauge': ['Site_R80_C1', 'Site_R80_C2', 'Site_R80_C3', 'Site_R80_C4', 'Site_R80_C6', 'Site_R81_C1', 'Site_R81_C2', 'Site_R81_C3', 'Site_R81_C6', 'Site_R82_C1', 'Site_R82_C2', 'Site_R82_C5', 'Site_R82_C6', 'Site_R83_C1', 'Site_R83_C3', 'Site_R83_C4', 'Site_R83_C6'],
'plot_cd': ['131', '141', '151', '161', '191', '221', '251', '281', '431', '131', '140', '180', '210', '240', '320', '380', '510'],
'draw_cd': ['60.8', '65', '69.5', '74', '92.4', '293', '297', '285', '278', '60.8', '62.4', '67.6', '65.4', '59.2', '68', '62', '59']}

gauges={'tone_sgn': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'model_cd': [81.2, 81.18, 80.12, 79.69, 77.06, 72.9, 69.44, 69.23, 68.29, 67.02, 65.42, 64.77, 64.86, 65.09, 65.08, 64.51, 63.23, 63.4, 63.63, 63.93, 0.0], 'range_min': [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0], 'range_max': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 'model_error': [-1.02, -1.47, -14.2, -1.61, -1.58, -0.38, 0.54, -0.3, -0.35, -0.22, -0.58, -1.37, -0.92, -0.81, -0.07, -0.31, -0.62, -0.55, -0.35, -0.92, 0.0], 'wafer_cd': [82.22, 82.65, 94.32, 81.3, 78.64, 73.28, 68.9, 69.53, 68.64, 67.24, 66.0, 66.14, 65.78, 65.9, 65.15, 64.82, 63.85, 63.95, 63.98, 64.85, 89.54], 'gauge': ['Site_R133_C3', 'Site_R133_C4', 'Site_R133_C6', 'Site_R134_C1', 'Site_R134_C2', 'Site_R134_C3', 'Site_R134_C4', 'Site_R134_C5', 'Site_R134_C6', 'Site_R135_C1', 'Site_R135_C2', 'Site_R135_C3', 'Site_R135_C4', 'Site_R135_C5', 'Site_R135_C6', 'Site_R136_C1', 'Site_R136_C2', 'Site_R136_C3', 'Site_R136_C4', 'Site_R136_C5', 'Site_R136_C6'], 'cost_wt': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0], 'plot_cd': [150.0, 155.0, 165.0, 170.0, 180.0, 200.0, 220.0, 240.0, 260.0, 290.0, 320.0, 340.0, 360.0, 400.0, 420.0, 460.0, 520.0, 580.0, 680.0, 880.0, 1280.0], 'draw_cd': [70.8, 81.0, 81.0, 78.2, 81.8, 87.2, 70.0, 70.0, 92.3, 67.0, 87.6, 66.0, 66.0, 88.0, 65.0, 65.0, 64.0, 82.6, 64.0, 64.0, 83.6]}

def sort_by_key(gauges, sort_key):
    if not gauges.has_key(sort_key):
        print("Can't found the key", sort_key)
        return gauges    
    length = len(gauges[sort_key])    
    for i in range(length) :
        swaped = False
        for j in range(length-1, i, -1):
            if gauges[sort_key][j-1] > gauges[sort_key][j]:
                swaped = True
                for key in gauges:
                    temp = gauges[key][j-1]
                    gauges[key][j-1] = gauges[key][j]
                    gauges[key][j] = temp
        if swaped == False:
            break
    return gauges
    
# when iteretally removing some items by certain criteria
# It's better to do it reversely, and the condition must be if-elif 
def gauge_filter(gauges):   
    length = len(gauges['gauge'])
    del_num = 0
    i = 0
    for i in xrange(length - 1, -1, -1):
        # filter 1: cost_wt<0.000001
        if gauges.has_key("cost_wt") and float(gauges['cost_wt'][i]) < 0.000001:
            print("Delete {} for cost_wt <= 0".format(gauges['gauge'][i]))
            del_num = del_num + 1
            for key in gauges:
                del gauges[key][i]
        # filter 2: wafer_cd<0.00001
        elif gauges.has_key("wafer_cd") and float(gauges['wafer_cd'][i]) < 0.000001:
            print("Delete {} for wafer_cd <= 0".format(gauges['gauge'][i]))
            del_num = del_num + 1
            for key in gauges:
                del gauges[key][i]
    print("Delete {} gauges in gauge_filter".format(del_num))
    return gauges

print(len(gauges['gauge']))
gauges = gauge_filter(gauges)
gauges = sort_by_key(gauges, 'draw_cd')
gauges = sort_by_key(gauges, 'plot_cd')
print(gauges)