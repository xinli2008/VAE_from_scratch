import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, hiddens = None, latent_dim = 128) -> None:
        """
        Variational Autoencoder model.

        Args:
            hiddens(list): List of hidden layer dimensions for the encoder and decoder.
            latent_dim(int): Dimensionality of the latent space.
        """
        super().__init__()

        if hiddens is None:
            hiddens = [16, 32, 64, 128, 256]

        self.latent_dim = latent_dim
        self.encoder, prev_channels, image_length = self._build_encoder(hiddens)
        self.mean_linear = nn.Linear(prev_channels * image_length * image_length, latent_dim)
        self.vae_linear  = nn.Linear(prev_channels * image_length * image_length, latent_dim)
        
        self.decoder_projection = nn.Linear(latent_dim, prev_channels * image_length * image_length)
        self.decoder_input_shape = (prev_channels, image_length, image_length)
        self.decoder = self._build_decoder(hiddens)

    
    def _build_encoder(self, hiddens):
        encoder_modules = []
        prev_channels = 3
        image_length = 64

        for cur_channel in hiddens:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels = prev_channels, out_channels = cur_channel, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(cur_channel),
                    nn.ReLU()
                )
            )
            prev_channels = cur_channel
            # NOTE:
            image_length = image_length // 2

        return nn.Sequential(*encoder_modules), prev_channels, image_length

    def _build_decoder(self, hiddens):
        decoder_modules = []
        for i in range(len(hiddens)-1, 0, -1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels = hiddens[i], out_channels = hiddens[i-1], kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(hiddens[i-1]),
                    nn.ReLU()
                )
            )
        
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0], hiddens[0], kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                nn.BatchNorm2d(hiddens[0]),
                nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, kernel_size = 3, stride = 1, padding = 1),
                nn.ReLU()
            )
        )
    
        return nn.Sequential(*decoder_modules)

    def forward(self, x):
        # forward encoderd
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)

        # latent z
        mean = self.mean_linear(encoded)
        var  = self.log_var_linear(encoded)
        eps = torch.randn_like(var)
        std = torch.exp(var / 2)
        z   = eps * std + mean
        
        # decoder
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *(self.decoder_input_shape)))
        decoded = self.decoder(x)

        return decoded, mean, var
    
    def sample(self, device = "cuda"):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input))
        decoded = self.decoder(x)
        return decoded     


if __name__ == "__main__":
    model = VAE()
    input_tensor = torch.randn([1, 3, 64, 64])
    output = model(input_tensor)
    for item in output:
        print(item.shape)