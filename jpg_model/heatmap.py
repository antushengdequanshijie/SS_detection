import torch
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_backward_hook(hook_function)

    def generate_cam(self, input_image, target_class):
        model_output = self.model(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())

        self.model.zero_grad()
        class_loss = model_output[0, target_class]
        class_loss.backward()

        guided_gradients = self.gradients.data.numpy()[0]
        target = self.target_layer.output[0].data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        cam = np.ones(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, input_image.shape[2:])
        return cam


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image


def main(image_path):
    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[1].conv2
    grad_cam = GradCAM(model, target_layer)

    image = load_image(image_path)
    cam = grad_cam.generate_cam(image, None)

    plt.imshow(cam, cmap='hot', interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    main("path_to_your_image.jpg")
