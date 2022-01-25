import torch
import punet

# load a test parcellation image
item = torch.load('item.pt')

# instantiate the pretrained model
image = item[0][None]
model = punet.unet2d_320_dktatlas_positional_20_1_0_0()

# inference
output = model(image)[0].argmax(0)

# pixel accuracy
gtruth = item[1].float().argmax(0)
acc = torch.mean((gtruth == output).float())
