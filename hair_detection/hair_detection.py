import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from MobileNetV2_unet import MobileNetV2_unet

# [226, 161, 255],  ##FFA1E2,
# [252, 54, 255],  ##FF36FC,


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


def map_colors(value):
    value = int(value)
    value = max(0, value)
    value = min(9, value)
    return color_map[value]


# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet()
    state_dict = torch.load("hair_detection/model/model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":

    model = load_model()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    # fig = plt.figure()

    image = cv2.imread("1.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)

    logits = model(torch_img)

    # print(logits.shape)  # 1, 3, 224, 224
    temp_logits = logits.squeeze().detach().numpy()

    # print(np.amax(temp_logits[0, :, :]))  # 14.204246
    # print(np.amin(temp_logits[0, :, :]))  # -6.9482875

    # print(np.amax(temp_logits[1, :, :]))  # 3.473512
    # print(np.amin(temp_logits[1, :, :]))  # -4.5498548

    # print(np.amax(temp_logits[2, :, :]))  # 9.963926
    # print(np.amin(temp_logits[2, :, :]))  # -6.030496

    # 12.734982
    # -8.217345
    # 3.6478317
    # -3.7493665
    # 10.566771
    # -8.214586

    # 13.043546
    # -3.953794
    # 2.0014768
    # -4.19139
    # 9.470165
    # -7.055818

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

    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # output = cv2.filter2D(src=output, ddepth=-1, kernel=kernel)

    # cv2.imshow("image", output)
    # cv2.waitKey(0)

    cv2.imwrite("output_4_testing.jpg", output)

    # Plot
    # ax = plt.subplot(111)
    # ax.axis("off")
    # ax.imshow(image.squeeze())

    # ax = plt.subplot(111)
    # ax.axis("off")
    # ax.imshow(mask)

    # plt.show()

    # arr = []
    # for i in temp_logits[2, :, :]:
    #     for j in i:
    #         arr.append(j)

    # x = arr

    # # plotting
    # plt.title("Line graph")
    # plt.xlabel("X axis")
    # plt.plot(x, color="red")
    # plt.savefig("mygraph.png")
