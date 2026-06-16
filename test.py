from torchvision import transforms
from dataset import OCRDataset

transform = transforms.Compose([
    transforms.Resize((32, 120)),
    transforms.ToTensor(),
])

dataset = OCRDataset("data/train", transform=transform)

print("样本数：", len(dataset))
img, text = dataset[0]
print("图片 shape:", img.shape)
print("标签:", text)
