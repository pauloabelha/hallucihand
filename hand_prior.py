import numpy as np
import sys
import muellericcv2017_importer

synthhands_handler, egodexter_handler = muellericcv2017_importer.load()

NUM_HANDJOINT_JOINTS = 21
NUM_DIMS = 3

RES_POS = 1 # in mm

MIN_POS = 0
MAX_POS = 250

NUM_DIFF_POS = len(range(MIN_POS, MAX_POS, RES_POS))
NUM_DIFF_JOINT_POS = NUM_DIFF_POS * NUM_HANDJOINT_JOINTS * NUM_DIMS
print("Num diff positions: " + str(NUM_DIFF_POS))

def get_num_dims(num_params):
    return int(((num_params * num_params) - num_params) / 2)

hand_prior = np.zeros((NUM_HANDJOINT_JOINTS * NUM_DIMS, NUM_DIFF_POS))
SIZE_PRIOR_GIGABYTES = sys.getsizeof(hand_prior) / (1024 * 1024 * 1024)
print("Size of prior (GB): " + str(SIZE_PRIOR_GIGABYTES))


ROOT_FOLDER_ERGODEXTER = '/home/paulo/rds_muri/paulo/EgoDexter/'
train_loader_ergodexter = egodexter_handler.get_loader(type='train',
                                            root_folder=ROOT_FOLDER_ERGODEXTER,
                                            img_res=(640, 480),
                                            batch_size=1,
                                            verbose=True)

ROOT_FOLDER_SYNTHHANDS = '/home/paulo/rds_muri/paulo/synthhands/SynthHands_Release/'
train_loader_synthhands = synthhands_handler.get_SynthHands_trainloader(root_folder=ROOT_FOLDER_SYNTHHANDS,
                                                             joint_ixs=range(21),
                                                             heatmap_res=(640, 480),
                                                             batch_size=1,
                                                             verbose=True)

min_joints = [1e10, 1e10, 1e10]
max_joints = [-1e10, -1e10, -1e10]

for batch_idx, (data, target) in enumerate(train_loader_synthhands):
    _, target_joints, _ = target
    joints = target_joints.numpy().reshape((21, 3))
    for joint_ix in range(21):
        joint = joints[joint_ix, :]
        joint_rel = joint - joints[0, :]
        joint_rel_discr = list(map(int, joint_rel))
    aa = 0


