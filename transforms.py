import torchvision as tv
import torchvision.transforms.v2 as v2





default = tv.transforms.Compose([
        v2.RandomPerspective(distortion_scale=0.3, p=0.5),
        v2.RandomGrayscale(p=0.1),
        v2.RandomRotation(degrees=(0, 15)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3),
    ])

autoaugment = tv.transforms.Compose([
        v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
    ])

autoaugmentImageNet = tv.transforms.Compose([
        v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
    ])


# More intense data augmentation to enforce regularization
hardAugmentation = tv.transforms.Compose([
    v2.RandomPerspective(distortion_scale=0.6, p=0.5),
    v2.RandomGrayscale(p=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomVerticalFlip(p=0.1),
    # v2.RandomInvert(p=0.5),
    v2.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3),
])

