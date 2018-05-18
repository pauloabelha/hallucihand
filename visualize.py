import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

def plot_3D_joints(joints_vec, ax=None, fig=None):
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    joints_vec = joints_vec.reshape((21, 3))
    for i in range(5):
        idx = (i * 4) + 1
        ax.plot([joints_vec[0, 1], joints_vec[idx, 1]],
                [joints_vec[0, 0], joints_vec[idx, 0]],
                [joints_vec[0, 2], joints_vec[idx, 2]],
                label='',
                color='C0')
    for j in range(5):
        idx = (j * 4) + 1
        for i in range(3):
            ax.plot([joints_vec[idx, 1], joints_vec[idx + 1, 1]],
                    [joints_vec[idx, 0], joints_vec[idx + 1, 0]],
                    [joints_vec[idx, 2], joints_vec[idx + 1, 2]],
                    label='',
                    color='C' + str(j+1))
            idx += 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.set_zlim(0, 500)
    ax.view_init(azim=270, elev=250)
    return ax, fig

def show():
    plt.show()