import torch.nn as nn

NUM_JOINTS = 21
LATENT_SPACE_PRE_MAP_DIMS = 40
LATENT_SPACE_DIMS = 23

class VAEHand_Encoder(nn.Module):

    def __init__(self):
        super(VAEHand_Encoder, self).__init__()

        self.fc1 = nn.Linear(in_features=NUM_JOINTS*3,
                             out_features=LATENT_SPACE_PRE_MAP_DIMS)
        self.fc2 = nn.Linear(in_features=LATENT_SPACE_PRE_MAP_DIMS,
                             out_features=LATENT_SPACE_DIMS)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class VAEHand_Decoder(nn.Module):

    def __init__(self):
        super(VAEHand_Decoder, self).__init__()

        self.fc1 = nn.Linear(in_features=LATENT_SPACE_DIMS,
                             out_features=LATENT_SPACE_PRE_MAP_DIMS)
        self.fc2 = nn.Linear(in_features=LATENT_SPACE_PRE_MAP_DIMS,
                             out_features=NUM_JOINTS*3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

class VAEHand(nn.Module):


    def __init__(self):
        super(VAEHand, self).__init__()

        self.encoder = VAEHand_Encoder()
        self.decoder = VAEHand_Decoder()

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x