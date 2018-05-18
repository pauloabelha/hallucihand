import torch.nn as nn

class VAEHand_Encoder(nn.Module):

    def __init__(self, num_joints=21, latent_premap_dims=40, latent_dims=23):
        super(VAEHand_Encoder, self).__init__()

        self.fc1 = nn.Linear(in_features=num_joints*3,
                             out_features=latent_premap_dims)
        self.fc2 = nn.Linear(in_features=latent_premap_dims,
                             out_features=latent_dims)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class VAEHand_Decoder(nn.Module):

    def __init__(self, num_joints=21, latent_premap_dims=40, latent_dims=23):
        super(VAEHand_Decoder, self).__init__()

        self.fc1 = nn.Linear(in_features=latent_dims,
                             out_features=latent_premap_dims)
        self.fc2 = nn.Linear(in_features=latent_premap_dims,
                             out_features=num_joints*3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

class VAEHand(nn.Module):


    def __init__(self, num_joints=21, latent_premap_dims=40, latent_dims=23):
        super(VAEHand, self).__init__()

        self.encoder = VAEHand_Encoder(num_joints, latent_premap_dims, latent_dims)
        self.decoder = VAEHand_Decoder(num_joints, latent_premap_dims, latent_dims)

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x