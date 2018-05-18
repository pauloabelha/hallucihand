import load_muellericcv2017_handlers
from torch.autograd import Variable
from vae_hand import VAEHand
import torch
import argparse
import matplotlib.pyplot as plt
from losses import euclidean_loss
import visualize

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-c', dest='checkpoint_filepath', default='',
                    help='Checkpoint file from which to begin training')
parser.add_argument('--log_interval', type=int, dest='log_interval', default=10,
                    help='Number of iterations interval on which to log'
                         ' a model checkpoint (default 10)')
parser.add_argument('--latent_dims', type=int, dest='latent_dims', default=23,
                    help='Number of latent dims for VAE')
parser.add_argument('--cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda for training')
parser.add_argument('--verbose', dest='verbose', action='store_true', default=True,
                    help='Whether to use cuda for training')
parser.add_argument('-o', dest='output_filepath', default='',
                    help='Output file for logging')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=16,
                    help='Batch size for training (if larger than max memory batch, training will take '
                         'the required amount of iterations to complete a batch')
parser.add_argument('-r', dest='root_folder', default='', required=True, help='Root folder for dataset')
args = parser.parse_args()




synthhands_handler, _ = load_muellericcv2017_handlers.load()

def load_checkpoint(filename, use_cuda=True):
    torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
    '''
    if use_cuda:
        try:
            torch_file = torch.load(filename)
        except:
            torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
            use_cuda = False
    else:
        torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
    '''
    model_state_dict = torch_file['model_state_dict']
    train_vars = torch_file['train_vars']
    params_dict = {}
    if not use_cuda:
        params_dict['use_cuda'] = False
    latent_dims = 23
    try:
        latent_dims = train_vars['latent_dims']
    except:
        pass
    model = VAEHand(latent_dims=latent_dims)
    model.load_state_dict(model_state_dict)
    if use_cuda:
        model = model.cuda()
    optimizer_state_dict = torch_file['optimizer_state_dict']
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(optimizer_state_dict)
    del optimizer_state_dict, model_state_dict
    return model, optimizer, train_vars


def get_losses_from_file_suffix(file_suffix, root_folder='/home/paulo/hallucihand/output'):
    losses = []
    if file_suffix == '':
        filepath = root_folder + '.txt'
    else:
        filepath = root_folder + '_' + file_suffix + '.txt'
    with open(filepath) as f:
        for line in f:
            loss_ix = line.find('Loss ')
            loss = float(line[loss_ix + 5:]) / 21.0
            losses.append(loss)
    return losses

vaehand, optimizer, train_vars = load_checkpoint('/home/paulo/hallucihand/vae_hand_log_23.pth.tar', use_cuda=False)

train_vars = {}
train_vars['root_folder'] = args.root_folder
train_vars['batch_size'] = args.batch_size
train_vars['log_interval'] = args.log_interval
train_vars['use_cuda'] = args.use_cuda
train_vars['output_filepath'] = args.output_filepath
train_vars['latent_dims'] = args.latent_dims
train_vars['verbose'] = args.verbose
train_vars['losses'] = []
train_vars['avg_losses'] = []
train_vars['total_loss'] = 0
train_vars['epoch'] = 0
train_vars['iter'] = 0
train_vars['batch_idx'] = 0


suffixes = ['5', '10', '23', '30', '31']
#suffixes = ['23', '30', '31']
losses_list = []
min_len = 1e10
for i in range(len(suffixes)):
    losses = get_losses_from_file_suffix(file_suffix=suffixes[i])
    if len(losses) < min_len:
        min_len = len(losses)
    losses_list.append(losses)

min_ix = 10
max_ix = min_len
handles = []
for i in range(len(suffixes)):
    handle = plt.plot(losses_list[i][min_ix:max_ix], label=suffixes[i])
    handles.append(handle)
plt.legend()
plt.show()


train_loader_synthhands = synthhands_handler.get_SynthHands_validloader(root_folder=train_vars['root_folder'],
                                                             joint_ixs=range(21),
                                                             heatmap_res=(640, 480),
                                                             batch_size=1,
                                                             verbose=True)

for batch_idx, (data, target) in enumerate(train_loader_synthhands):
    train_vars['batch_idx'] = batch_idx
    _, target_joints_orig, _ = target
    target_joints = Variable(target_joints_orig)
    if train_vars['use_cuda']:
        target_joints = target_joints.cuda()
    output_joints = vaehand(target_joints)
    loss = euclidean_loss(output_joints, target_joints)
    print(loss.item())
    ax3d, fig = visualize.plot_3D_joints(target_joints_orig)
    visualize.plot_3D_joints(output_joints, ax=ax3d, fig=fig)
    visualize.show()
    aa = 0
