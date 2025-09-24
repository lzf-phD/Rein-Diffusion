# Rein-Diffusion
A Structure-Guided Diffusion Framework for Automated Reinforcement Design of Bridge RC Piles

![graph abstract.png](other/abstract.png)

# Lib Name-Version
python                    3.6.13  
pytorch                   1.7.1   
numpy                     1.19.2

# dataset
1. For training sets of RC pile reinforcement drawings: 
   - `Rein/train_B` store the **labels**.  
   - `Rein/mask` store the **mask maps**.  
   - `Rein/cond` store the **text conditions**.
2. For testing sets  of RC pile reinforcement drawings:   
   - `Test/test_B` store the **labels**.  
   - `Test/mask` store the **mask maps**.  
   - `Test/cond` store the **text conditions**.
3. Model prediction results (outputs):  
   - `rein/samples_controlnet` store the different diameters of RC pile reinforcement drawings.

**Note: Due to project privacy and intellectual property considerations, only a representative subset of the training dataset is publicly shared for reference.**

# Datasets directory
data/  
├── Rein/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Training dataset   
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_A/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_B/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# labels  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── mask/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# mask maps  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── cond/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# text conditions  
└── Test/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Testing dataset  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_A/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_B/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# labels  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── mask/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# mask maps  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── cond/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# text conditions  


# Training
To train variational autoencoder(vae).
```bash
   python train_vae.py
```
To train latent diffusion model(ldm).
```bash
   python train_ldm_vae.py
```
To train controlnet.
```bash
   python train_ldm_controlnet.py
```
# Inference
The trained model weight files (.pth) are stored in the checkpoint directory.

To test variational autoencoder(vae).
```bash
   python infer_vae.py
```
To test latent diffusion model(ldm).
```bash
   python sample_ldm_vae.py
```
To test controlnet.
```bash
   python sample_ldm_controlnet.py
```

# Visualized result
Partial visualized test results are available in the directory as follows.
   - `rein/samples_controlnet_1.0d` for 1.0m diameters of RC pile reinforcement drawing.
   - `rein/samples_controlnet_1.2d` for 1.2m diameters of RC pile reinforcement drawing.
   - `rein/samples_controlnet_1.5d` for 1.5m diameters of RC pile reinforcement drawing.
   - `rein/samples_controlnet_1.8d` for 1.8m diameters of RC pile reinforcement drawing.
   - `rein/samples_controlnet_2.0d` for 2.0m diameters of RC pile reinforcement drawing.