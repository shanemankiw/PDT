import numpy as np

def parse_joints_info(txt_path):
    """
    Input should be the rig_info txt from RigNet
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    joints = []
    for line in lines:
        if line.startswith('joints'):
            joint = line.split(' ')[2:]
            joint = [float(j) for j in joint]
            joints.append(np.array(joint))

    return joints