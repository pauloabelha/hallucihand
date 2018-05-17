import numpy as np
import sys

NUM_HANDJOINT_ANGLES = 25
NUM_HANDROOT_ANGLES = 3
NUM_HANDROOT_POS = 3

RES_ANGLE = 1 # in degrees
RES_POS = 1 # in mm

MIN_ANGLE = 0
MAX_ANGLE = 360
MIN_POS = 0
MAX_POS = 300

NUM_DIFF_ANGLES = len(range(MIN_ANGLE, MAX_ANGLE, RES_ANGLE)) * (NUM_HANDJOINT_ANGLES + NUM_HANDROOT_ANGLES)
print("Num angles: " + str(NUM_DIFF_ANGLES))

NUM_DIFF_POS = len(range(MIN_POS, MAX_POS, RES_POS)) * (NUM_HANDROOT_POS)
print("Num positions: " + str(NUM_DIFF_POS))

NUM_DIFF_PARAMS = NUM_DIFF_ANGLES + NUM_DIFF_POS
print("Num diff params: " + str(NUM_DIFF_PARAMS))

def get_num_dims(num_params):
    return int(((num_params * num_params) - num_params) / 2)

#NUM_ELEMS_PRIOR = get_num_dims(NUM_DIFF_PARAMS)
#print("Num elems in prior: " + str(NUM_ELEMS_PRIOR))

hand_prior = np.zeros((NUM_DIFF_PARAMS, NUM_DIFF_PARAMS))
SIZE_PRIOR_GIGABYTES = sys.getsizeof(hand_prior) / (1024 * 1024 * 1024)
print("Size of prior (GB): " + str(SIZE_PRIOR_GIGABYTES))