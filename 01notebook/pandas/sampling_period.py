import numpy as np
import os, os.path
import pandas as pd

workpath = os.path.dirname(os.path.abspath(__file__))
period_file = os.path.join(workpath, "period.txt")

df = pd.read_csv(period_file, delimiter = '\s+')
list_period = df.columns.values.tolist() 

#hexc = df["Hex_code"].values.tolist()
print(list_period)
print(len(list_period), df)
print(len(list_period), list_period)

# Xperiod
# BWALL__CHass_NoOPC = [150, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200]
# BWALL__CH_AR__CH  = [172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252, 260, 268, 276, 284, 292, 300, 320, 360, 400, 480, 560, 600, 800, 1000, 1200]
# BWALL__CH_AR__Row = [180, 184, 192, 200, 208, 216, 224, 232, 240, 260, 280, 300, 320, 340, 360, 380, 400]
# BWALL__CH_Stag_AR__CH_Stag = [110, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
# 

# Yperiod
# BWALL__CH_AR__CHE   = [90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 400, 500, 600]

# P5
# Cut_01_NoOPC = [388, 390, 394, 398, 402, 406, 410, 414, 418, 422, 426, 430, 434, 438, 442, 446, 450, 454, 458, 462, 466, 470, 474, 478, 482, 486]
# Cut_SAQP_01_NoOPC = [384, 388, 390, 394, 398, 402, 406, 410, 414, 418, 422, 426, 430, 434, 438, 442, 446, 450]
# Block_SAQP_01_NoOPC = [384, 386, 390, 394, 398, 402, 406, 410, 414, 418, 422, 426, 430, 434, 438, 442, 446, 450]

# p0 
# Keep_Fin_01_NoOPC = [90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
# Keep_M_01_NoOPC = [90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]

# rest small module
# module_list = ['DenseLinearity_CUTMASK_01A', 'IsoLinearity_CUTMASK_01A', 'MODULE_CH_Bite', 'MODULE_CH_Stag_yongfa']