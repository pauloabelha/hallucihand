import muellericcv2017_importer
from torch.autograd import Variable
import numpy as np
from losses import euclidean_loss
from vae_hand import VAEHand
import torch.optim as optim
import torch
import argparse

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, required=True,
                    help='Total number of iterations to train')
parser.add_argument('--rel_pos', dest='rel_pos', action='store_true', default=False,
                    help='Whether to use relative joint position (to the hand root)')
parser.add_argument('--log_interval', type=int, dest='log_interval', default=10,
                    help='Number of iterations interval on which to log'
                         ' a model checkpoint (default 10)')
parser.add_argument('--latent_dims', type=int, dest='latent_dims', default=23,
                    help='Number of latent dims for VAE')
parser.add_argument('--cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda for training')
parser.add_argument('--verbose', dest='verbose', action='store_true', default=True,
                    help='Whether to use cuda for training')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=16,
                    help='Batch size for training (if larger than max memory batch, training will take '
                         'the required amount of iterations to complete a batch')
parser.add_argument('-r', dest='root_folder', default='', required=True, help='Root folder for dataset')
args = parser.parse_args()


train_vars = {}
train_vars['root_folder'] = args.root_folder
train_vars['latent_dims'] = args.latent_dims
train_vars['rel_pos'] = args.rel_pos
if train_vars['rel_pos']:
    rel_pos_str = '_relpos'
else:
    rel_pos_str = ''
output_filenamebase = 'vaehand_' + str(train_vars['latent_dims']) + rel_pos_str
train_vars['checkpoint_filepath'] = output_filenamebase + '.pth.tar'
train_vars['output_filepath'] = output_filenamebase + '.txt'
train_vars['num_epochs'] = args.num_epochs
train_vars['batch_size'] = args.batch_size
train_vars['log_interval'] = args.log_interval
train_vars['use_cuda'] = args.use_cuda

train_vars['latent_dims'] = args.latent_dims
train_vars['verbose'] = args.verbose
train_vars['losses'] = []
train_vars['avg_losses'] = []
train_vars['total_loss'] = 0
train_vars['epoch'] = 0
train_vars['iter'] = 0
train_vars['batch_idx'] = 0

def print_verbose(str, verbose, n_tabs=0, erase_line=False):
    prefix = '\t' * n_tabs
    msg = prefix + str
    if verbose:
        if erase_line:
            print(msg, end='')
        else:
            print(prefix + str)
    return msg


loaded_modules = muellericcv2017_importer.load_modules()
synthhands_handler = loaded_modules['synthhands_handler']

train_loader_synthhands = synthhands_handler.get_SynthHands_trainloader(root_folder=train_vars['root_folder'],
                                                             joint_ixs=range(21),
                                                             heatmap_res=(640, 480),
                                                             batch_size=train_vars['batch_size'],
                                                             verbose=True)

len_dataset=len(train_loader_synthhands.dataset)
if train_vars['rel_pos']:
    vaehand = VAEHand(num_joints=20, latent_dims=train_vars['latent_dims'])
else:
    vaehand = VAEHand(latent_dims=train_vars['latent_dims'])
adam_optim = optim.Adam(params=vaehand.parameters(), lr=0.001)

if train_vars['use_cuda']:
    vaehand = vaehand.cuda()

for epoch in range(train_vars['num_epochs']):
    train_vars['epoch'] = epoch
    for batch_idx, (data, target) in enumerate(train_loader_synthhands):
        train_vars['batch_idx'] = batch_idx
        _, target_joints, _ = target
        if train_vars['rel_pos']:
            target_joints_rel = np.zeros((target_joints.shape[0], target_joints.shape[1]-3))
            for i in range(target_joints.shape[0]):
                target_joints_reshaped = target_joints[i].reshape((21, 3))
                target_joints_reshaped -= target_joints_reshaped[0, :]
                target_joints_rel[i] = target_joints_reshaped[1:, :].reshape((60))
            target_joints = torch.from_numpy(target_joints_rel).float()
        target_joints = Variable(target_joints)
        if train_vars['use_cuda']:
            target_joints = target_joints.cuda()
        output_joints = vaehand(target_joints)
        loss = euclidean_loss(output_joints, target_joints)
        train_vars['losses'].append(loss.item())
        train_vars['total_loss'] += loss.item()
        loss.backward()
        adam_optim.step()
        adam_optim.zero_grad()
        if (train_vars['iter']+1) % train_vars['log_interval'] == 0:
            print('Saving model...')
            checkpoint_model_dict = {
                'model_state_dict': vaehand.state_dict(),
                'optimizer_state_dict': adam_optim.state_dict(),
                'train_vars': train_vars,
            }
            torch.save(checkpoint_model_dict, train_vars['checkpoint_filepath'])
            avg_loss = train_vars['total_loss'] / train_vars['log_interval']
            train_vars['avg_losses'].append(avg_loss)
            train_vars['total_loss'] = 0
            perc_batch_completed = int(100 * (train_vars['batch_idx'] *
                                              train_vars['batch_size']) /
                                       len_dataset)
            perc_epoch_completed = int(100 * train_vars['epoch'] /
                                       train_vars['num_epochs'])
            msg = ''
            msg += print_verbose('Epoch {}/{} : {}% | Batch {}({}) : {}% | Loss {}'.
                  format(train_vars['epoch']+1, train_vars['num_epochs'],
                         perc_epoch_completed, (train_vars['batch_idx']+1),
                         train_vars['batch_size'], perc_batch_completed,
                         avg_loss), train_vars['verbose'])
            if not train_vars['output_filepath'] == '':
                with open(train_vars['output_filepath'], 'a') as f:
                    f.write(msg + '\n')
        train_vars['iter'] += 1
