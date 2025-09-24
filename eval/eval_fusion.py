import torch
from PIL import Image
import torchvision.transforms as transforms


path1 = "label_img.png"
path2 = "output_0.png"
to_tensor = transforms.ToTensor()

img1 = Image.open(path1).convert("RGB")
img2 = Image.open(path2).convert("RGB")

tensor1 = (to_tensor(img1).permute(1, 2, 0) * 255).byte()
tensor2 = (to_tensor(img2).permute(1, 2, 0) * 255).byte()

color_map = {
    (255, 255, 255): 0,
    (255, 0, 0): 1,
    (132, 132, 132): 2
}

H, W, _ = tensor1.shape
labels1 = torch.full((H, W), -1, dtype=torch.long)
labels2 = torch.full((H, W), -1, dtype=torch.long)

for rgb, idx in color_map.items():
    mask1 = (tensor1 == torch.tensor(rgb, dtype=torch.uint8)).all(dim=2)
    mask2 = (tensor2 == torch.tensor(rgb, dtype=torch.uint8)).all(dim=2)
    labels1[mask1] = idx
    labels2[mask2] = idx

y_true = labels1.view(-1)
y_pred = labels2.view(-1)

num_classes = len(color_map)
conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)

for t, p in zip(y_true, y_pred):
    conf_mat[t, p] += 1

print("Confusion Matrix:")
print(conf_mat)

accuracy = conf_mat.diag().sum().item() / conf_mat.sum().item()
print("Accuracy:", accuracy)



