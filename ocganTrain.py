"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIEStai
from __future__ import print_function

from option1 import Options
from lib.data import load_data
from ocgan.model import ocgan

##
# def main():
""" Training
"""

##
# ARGUMENTS
opt = Options().parse()
print(1)
##
# LOAD DATA
dataloader = load_data(opt)
print(2)
##
# LOAD MODEL
model = ocgan(opt, dataloader)
print(3)
##
# TRAIN MODEL
model.train()
print(4)


