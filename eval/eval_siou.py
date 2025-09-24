import torch
from PIL import Image
import torchvision.transforms as transforms

path1 = "label_img.jpg"
path2 = "output_0.png"

to_tensor = transforms.ToTensor()

img1 = Image.open(path1).convert("RGB")
img2 = Image.open(path2).convert("RGB")

tensor1 = (to_tensor(img1).permute(1, 2, 0) * 255).byte()
tensor2 = (to_tensor(img2).permute(1, 2, 0) * 255).byte()

red = torch.tensor([255, 0, 0], dtype=torch.uint8)

mask1 = (tensor1 == red).all(dim=2)  # [H,W] bool
mask2 = (tensor2 == red).all(dim=2)  # [H,W] bool

intersection = (mask1 & mask2).sum().item()

total_red_tensor2 = mask2.sum().item()

ratio = intersection / total_red_tensor2 if total_red_tensor2 > 0 else 0.0

print(total_red_tensor2)
print(intersection)
print(ratio)

