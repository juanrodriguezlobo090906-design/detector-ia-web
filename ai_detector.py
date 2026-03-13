import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image

device = torch.device("cpu")

# modelo base
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def deep_model_prediction(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    prob = torch.sigmoid(output).item()

    return prob


def fft_analysis(image):

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)

    score = np.mean(magnitude) / 10

    score = min(score,1)

    return score


def patch_analysis(image):

    img = np.array(image)

    h,w,_ = img.shape

    patches = []

    size = 128

    for y in range(0,h-size,size):
        for x in range(0,w-size,size):

            patch = img[y:y+size,x:x+size]

            patch = Image.fromarray(patch)

            prob = deep_model_prediction(patch)

            patches.append(prob)

    if len(patches)==0:
        return 0.5

    return np.mean(patches)


def analyze_image(image_path):

    image = Image.open(image_path).convert("RGB")

    deep_prob = deep_model_prediction(image)

    fft_prob = fft_analysis(image)

    patch_prob = patch_analysis(image)

    final_score = (deep_prob*0.5) + (fft_prob*0.2) + (patch_prob*0.3)

    return round(final_score*100,2)