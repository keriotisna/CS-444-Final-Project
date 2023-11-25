import os
from pandas import read_csv
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader, random_split
from icecream import ic
from PIL import Image
import torchvision.transforms.v2 as v2

class CIFAR10Dataset(Dataset):
    
    # The defaultTransform is the first transform that should be called as it turns the inputs into a form
    # other transforms can work on.
    defaultTransform = v2.Compose([
        # Ensure everything is in the right size and format before ending our transforms
        # v2.Resize(size=(32, 32), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    def __init__(self, rootDirectory, csvFilename, dataFolder, transform:v2.Compose=None):
        
        
        csvPath = os.path.join(rootDirectory, csvFilename)
        annotations = read_csv(csvPath)
        
        dataPath = os.path.join(rootDirectory, dataFolder)
        
        # print(annotations.dtypes)
        
        # Convert labels to string https://stackoverflow.com/questions/22231592/pandas-change-data-type-of-series-to-string
        # annotations["label"] = annotations["label"].astype('string')
        # print(annotations.dtypes)

        # Replace string labels with integer labels 0-9
        unique = annotations['label'].unique()#.astype(str)
        stringNumberLabelEnumerations = enumerate(sorted(list(unique)))
        
        # Create dictionaries for easily mapping to and from strings or integers as picture labels
        stringNumberMappings = {string: number for number, string in stringNumberLabelEnumerations}
        numberStringMappings = {number: string for string, number in stringNumberMappings.items()}
        
        
        # .mask acts similarly to boolean indexing. If the condition is true, the current value is replaced with num+1
        for string, num in stringNumberMappings.items():
            annotations['label'].mask(annotations['label'] == string, num, inplace=True)        
        
        self.stringNumberMappings = stringNumberMappings
        self.numberStringMappings = numberStringMappings
        
        self.annotations = annotations
        self.dataPath = dataPath
        
        if transform is not None:
            # Combine compose transforms if another transform is given
            setTransform = v2.Compose(self.defaultTransform.transforms + transform.transforms)
        else:
            setTransform = self.defaultTransform
        
        self.transform = setTransform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        imageId = self.annotations.iloc[index, 0]
        imageName = os.path.join(self.dataPath, f"{imageId}.png")
        image = Image.open(imageName)
        
        label = self.annotations.iloc[index, 1]
        
        if self.transform:
            image = self.transform(image)

        return image, label












def main():


    transform = v2.Compose([
        tv.transforms.ToTensor(),
        # Add any other transformations you need
    ])

    # Create dataset instances
    fullDataset = CIFAR10Dataset(rootDirectory='cifar-10', csvFilename='trainLabels.csv', dataFolder='train', transform=transform)
    # test_dataset = CIFAR10LocalDataset(csv_file='path/to/testLabels.csv', root_dir='path/to/test', transform=transform)

    generator = torch.Generator().manual_seed(42)
    TRAIN_DATASET, VALIDATION_DATSET, TEST_DATSAET = random_split(fullDataset, [0.8, 0.1, 0.1], generator=generator)

    # Create DataLoader instances
    BATCH_SIZE = 256
    trainLoader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)

    train_features, train_labels = next(iter(trainLoader))
    
    trainFeaturesArray = train_features.numpy().transpose(2, 3, 1, 0)
    trainLabelsArray = train_labels.numpy()
    
    
    ic(train_features.size())
    ic(train_labels.size())


if __name__ == "__main__":
    main()