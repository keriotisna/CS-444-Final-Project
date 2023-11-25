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

NONE = v2.Compose([
    v2.Identity(),
])