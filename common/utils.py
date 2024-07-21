import torch
import albumentations as T
from albumentations.pytorch import ToTensorV2
import random
import torch.nn.init as init

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "path/to/your/train_folder/train"
VAL_DIR = "path/to/your/train_folder/val"
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 8
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_Y = "geny.pth.tar"
CHECKPOINT_GEN_O = "geno.pth.tar"
CHECKPOINT_CRITIC_Y = "criticy.pth.tar"
CHECKPOINT_CRITIC_O = "critico.pth.tar"

transforms = T.Compose(
    [
        T.Resize(width=256, height=256),
        T.HorizontalFlip(p=0.5),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)


def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
      
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]

    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    return epoch

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor):
        data = data.detach()
        res = []
        for element in data:
            if len(self.data) < self.max_size:
                self.data.append(element)
                res.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    res.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    res.append(element)
        return torch.stack(res)
    


