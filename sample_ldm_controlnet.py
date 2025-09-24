import torch
import torchvision
import argparse
import yaml
import os
import random
from torchvision.utils import make_grid
from tqdm import tqdm
from models.controlnet_ldm import ControlNet
from dataset.rein_dataset import ReinDataset
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, dataset):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((train_config['num_samples'],
                      autoencoder_model_config['z_channels'],
                      192,
                      96)).to(device)
    iodex=1
    origin_img = dataset[iodex][0] #img
    context = dataset[iodex][1].unsqueeze(0).to(device)  #txt_embedding
    hints = dataset[iodex][2].unsqueeze(0).to(device) #canny

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt,context,torch.as_tensor(i).unsqueeze(0).to(device), hints)

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.to(device).decode(xt)
        else:
            ims = xt

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        if not os.path.exists(os.path.join(train_config['task_name'], 'samples_controlnet')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples_controlnet'))
        img.save(os.path.join(train_config['task_name'], 'samples_controlnet', 'x0_{}.png'.format(i)))
        img.close()
    origin_img = (origin_img + 1) / 2
    label_img = torchvision.transforms.ToPILImage()(origin_img)
    label_img.save(os.path.join(train_config['task_name'], 'samples_controlnet', 'label_img.png'))
    label_img.close()


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'],
                                     ldm_scheduler=True)

    rein_canny = ReinDataset('test',
                               im_path=dataset_config['test_path'],
                               return_hint=True)

    # Instantiate the model
    downscale_factor = 8
    model = ControlNet(im_channels=autoencoder_model_config['z_channels'],
                       model_config=diffusion_model_config,
                       model_locked=True,
                       model_ckpt=os.path.join(train_config['task_name'], "ddpm_ckpt.pth"),
                       device=device,
                       down_sample_factor=downscale_factor).to(device)
    model.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       "ddpm_controlnet_ckpt_2200.pth")), "Train ControlNet first"
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  "ddpm_controlnet_ckpt_2200.pth"),
                                     map_location=device))
    print('Loaded controlnet checkpoint')

    vae = VAE(im_channels=dataset_config['im_channels'],
              model_config=autoencoder_model_config)
    vae.eval()

    # Load vae if found
    assert os.path.exists(os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])), \
        "VAE checkpoint not present. Train VAE first."
    vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                train_config['vae_autoencoder_ckpt_name']),
                                   map_location=device), strict=True)
    print('Loaded vae checkpoint')

    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, rein_canny)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ldm controlnet generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/rein.yaml', type=str)
    args = parser.parse_args()
    infer(args)
