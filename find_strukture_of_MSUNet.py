

from network.MSUNet import MSUNet as MSUNet
from config import get_config


config = get_config(None)
ps = config.MODEL.SWIN.PATCH_SIZE
ws = config.MODEL.SWIN.WINDOW_SIZE
img = config.DATA.IMG_SIZE
H = img // ps

model = MSUNet( config, 
                img_size = 1024, 
                num_classes = 1
                )
state_dict = model.state_dict()

# alle keys auflisten
for i, k in enumerate(state_dict.keys()):
    print(i, ":", k, state_dict[k].shape)
