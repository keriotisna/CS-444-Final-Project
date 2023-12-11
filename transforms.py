import torchvision as tv
import torchvision.transforms.v2 as v2





default = v2.Compose([
        v2.RandomPerspective(distortion_scale=0.4, p=0.5),
        v2.RandomGrayscale(p=0.2),
        v2.RandomRotation(degrees=(0, 15)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3),
    ])

autoaugment = v2.Compose([
        v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
    ])

autoaugmentImageNet = v2.Compose([
        v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
    ])


# More intense data augmentation to enforce regularization
hardAugmentation = v2.Compose([
    v2.RandomPerspective(distortion_scale=0.3, p=0.5),
    v2.RandomGrayscale(p=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.1),
    v2.RandomInvert(p=0.5),
    v2.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4),
])

easyaugmentation = v2.Compose([
        v2.RandomPerspective(distortion_scale=0.2, p=0.5),
        v2.RandomGrayscale(p=0.1),
        v2.RandomRotation(degrees=(0, 15)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3),
])

hardAugmentation2 = v2.Compose([
    v2.RandomPerspective(distortion_scale=0.4, p=0.5),
    v2.RandomGrayscale(p=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.2),
    v2.RandomInvert(p=0.2),
    v2.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4),
])

hardAugmentation3 = v2.Compose([
    v2.RandomPerspective(distortion_scale=0.3, p=0.5),
    v2.RandomGrayscale(p=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.2),
    v2.RandomInvert(p=0.2),
    v2.RandomResizedCrop(size=(32, 32), scale=(0.3, 1), antialias=True),
    v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4),
])


hardAugmentation2_5 = v2.Compose([
    # v2.RandomPerspective(distortion_scale=0.3, p=0.5),
    v2.RandomRotation(degrees=(0, 90)),
    v2.RandomGrayscale(p=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.1),
    v2.RandomInvert(p=0.1),
    v2.RandomResizedCrop(size=(32, 32), scale=(0.3, 1), antialias=True),
    v2.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3),
])

hardAugmentation2_6 = v2.Compose([
    v2.RandomPerspective(distortion_scale=0.4, p=0.5),
    v2.RandomGrayscale(p=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.3),
    v2.RandomInvert(p=0.2),
    v2.RandomResizedCrop(size=(32, 32), scale=(0.3, 1), antialias=True),
    v2.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4),
])



NONE = v2.Compose([
    v2.Identity(),
])

RESNET_18_NORMALIZATION = v2.Compose([
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) # tv.models.ResNet18_Weights.DEFAULT.transforms(), but converted to a v2.Compose