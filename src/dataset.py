import os
import torchvision.transforms as transforms
import torchvision


def read_voc_dataset(path, year, download):
    T = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            ])
    voc_data =  torchvision.datasets.VOCDetection(path, year=year, image_set='train', transform=T, download=download)
    voc_val =  torchvision.datasets.VOCDetection(path, year=year, image_set='val', transform=T, download=download)

    return voc_data, voc_val