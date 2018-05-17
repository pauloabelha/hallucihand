import load_muellericcv2017_handlers
from torch.autograd import Variable
from vae_hand import VAEHand
import torch.optim as optim
import torch

NUM_EPOCHS = 100
BATCH_SIZE = 16
LOG_INTERVAL = 10
USE_CUDA = True

train_vars = {}
train_vars['num_epochs'] = NUM_EPOCHS
train_vars['batch_size'] = BATCH_SIZE
train_vars['log_interval'] = LOG_INTERVAL
train_vars['losses'] = []
train_vars['avg_losses'] = []
train_vars['total_loss'] = 0
train_vars['epoch'] = 0
train_vars['iter'] = 0
train_vars['batch_idx'] = 0
train_vars['cuda'] = USE_CUDA
train_vars['checkpoint_filepath'] = 'vae_hand_log.pth.tar'

def euclidean_loss(output, target):
    batch_size = output.shape[0]
    return (output - target).abs().sum() / batch_size

synthhands_handler, _ = load_muellericcv2017_handlers.load()

ROOT_FOLDER_SYNTHHANDS = '/home/paulo/rds_muri/paulo/synthhands/SynthHands_Release/'
train_loader_synthhands = synthhands_handler.get_SynthHands_trainloader(root_folder=ROOT_FOLDER_SYNTHHANDS,
                                                             joint_ixs=range(21),
                                                             heatmap_res=(640, 480),
                                                             batch_size=BATCH_SIZE,
                                                             verbose=True)

len_dataset=len(train_loader_synthhands.dataset)
vaehand = VAEHand()
adam_optim = optim.Adam(params=vaehand.parameters(), lr=0.001)

if train_vars['cuda']:
    vaehand = vaehand.cuda()

for epoch in range(NUM_EPOCHS):
    train_vars['epoch'] = epoch
    for batch_idx, (data, target) in enumerate(train_loader_synthhands):
        train_vars['batch_idx'] = batch_idx
        _, target_joints, _ = target
        target_joints = Variable(target_joints)
        output_joints = vaehand(target_joints)
        loss = euclidean_loss(output_joints, target_joints)
        train_vars['losses'].append(loss.item())
        train_vars['total_loss'] += loss.item()
        loss.backward()
        adam_optim.step()
        adam_optim.zero_grad()
        if (train_vars['iter']+1) % LOG_INTERVAL == 0:
            print('Saving model...')
            checkpoint_model_dict = {
                'model_state_dict': vaehand.state_dict(),
                'optimizer_state_dict': adam_optim.state_dict(),
                'train_vars': train_vars,
            }
            torch.save(checkpoint_model_dict, train_vars['checkpoint_filepath'])
            avg_loss = train_vars['total_loss'] / LOG_INTERVAL
            train_vars['avg_losses'].append(avg_loss)
            train_vars['total_loss'] = 0
            perc_batch_completed = int(100 * (train_vars['batch_idx'] *
                                              train_vars['batch_size']) /
                                       len_dataset)
            perc_epoch_completed = int(100 * train_vars['epoch'] /
                                       train_vars['num_epochs'])
            print('Epoch {}/{} : {}% | Batch {}({}) : {}% | Loss {}'.
                  format(train_vars['epoch']+1, train_vars['num_epochs'],
                         perc_epoch_completed, (train_vars['batch_idx']+1),
                         train_vars['batch_size'], perc_batch_completed,
                         avg_loss))
        train_vars['iter'] += 1
