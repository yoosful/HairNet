import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from MobileNetV2_unet import MobileNetV2_unet


color_map = [
    [168, 241, 255],  ##FFF1A8
    [152, 250, 255],  ##FFFA99
    [88, 242, 255],  ##FFF258,
    [76, 233, 255],  ##FFE94C
    [62, 219, 255],  ##FFDB3E,
    [44, 192, 255],  ##FFC02C
    [24, 148, 255],  ##FF9418
    [16, 123, 255],  ##FF7B10,
    [7, 82, 255],  ##FF5207
    [0, 8, 255],  ##FF0800,
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--map_color", type=bool, default=False)
    parser.add_argument("--save_img", type=bool, default=False)
    return parser.parse_args()


def map_colors(value):
    value = int(value)
    value = max(0, value)
    value = min(9, value)
    return color_map[value]


def load_model():
    model = MobileNetV2_unet()
    state_dict = torch.load("hair_detection/model/model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    model = load_model()

    imagePath = get_args().image_path
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)

    logits = model(torch_img)

    temp_logits = logits.squeeze().detach().numpy()

    mask = np.argmax(logits.data.cpu().numpy(), axis=1)
    mask = mask.squeeze()

    mask_n = np.zeros((224, 224, 3))

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] == 1:  # Face
                mask_n[i][j] = [0, 0, 128]
                continue
            if mask[i][j] == 2:  # Hair
                mask_n[i][j] = map_colors(temp_logits[2, i, j])
                continue
            if mask[i][j] == 0:  # Background
                mask_n[i][j] = [0, 0, 0]
                continue

    output = mask_n.astype(np.uint8)
    output = cv2.resize(
        mask_n.astype(np.uint8), (128, 128), interpolation=cv2.INTER_LINEAR
    )

    if get_args().save_img == True:
        cv2.imwrite("output.png", output)

    ax = plt.subplot(121)
    ax.axis("off")
    ax.imshow(image.squeeze())

    ax = plt.subplot(122)
    ax.axis("off")
    if get_args().map_color == True:
        ax.imshow(output)
    else:
        ax.imshow(mask)

    plt.show()
