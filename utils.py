from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]

train_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        A.PadIfNeeded(min_height=4, min_width=4, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        ToTensorV2(),
    ]
)

class CIFAR10WithAlbumentations(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        
    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return image, label
    

def get_gradcam(model, use_cuda):
    target_layers = [model.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    return cam

def visualize_cam(cam, rgb_img, input_tensor, target, img_id):
    targets = [ClassifierOutputTarget(target)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization


def unnormalize(img):
    img = img / 2 + 0.5   # unnormalize
    npimg = img.cpu().numpy()   # convert from tensor
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)
  

def get_image(tensor):
    img = torchvision.utils.make_grid(tensor)
    img = img.cpu().numpy()
    img = interval_mapping(img, np.min(img), np.max(img), 0, 255)
    unnorm_img = np.transpose(img, (1, 2, 0))
    return unnorm_img


def get_misclassified_images_with_label(tensor, pred_label, class_to_idx):
    img = get_image(tensor)
    pred_class = ""
    for cls, idx in class_to_idx.items():
        if idx == pred_label:
            pred_class = cls
    return {
        "img": img,
        "pred_class": pred_class,
        "tensor": tensor,
        "pred_idx": pred_label
    }

def plot_misclassified_grad_cam_images(model, use_cuda, misclassified_images):
    cam = get_gradcam(model, use_cuda)
    f, axarr = plt.subplots(5,2, figsize=(8, 12))
    for i, miscl in enumerate(misclassified_images):
        f.add_subplot(5, 2, i+1)
        img = visualize_cam(cam, miscl["img"]/255, miscl["tensor"], miscl['pred_idx'], f"{miscl['pred_class']}_{i}")
        plt.imshow((img).astype(np.uint8))
        plt.xlabel(miscl['pred_class'], fontsize=15)
    f.tight_layout()
    plt.savefig("misclassified_grad_cam.png")
    plt.show()


def plot_misclassified(misclassified):
    f, axarr = plt.subplots(5,2, figsize=(8, 12))
    for num in range(1, 11):
        f.add_subplot(5, 2, num)
        idx = num - 1
        plt.imshow((misclassified[idx]["img"]).astype(np.uint8))
        plt.xlabel(misclassified[idx]["pred_class"], fontsize=15)

    f.tight_layout()
    plt.savefig("misclassified_images.png")
    plt.show()

def plot_losses(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    t = [t_items.item() for t_items in train_losses]
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")