import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import CRNN
from utils import CHARS, decode_ctc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="best_crnn.pth", help="Path to model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 120)),
        transforms.ToTensor(),
    ])

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    image = Image.open(args.image).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)              # [T, 1, C]
        pred = logits.argmax(dim=2).squeeze(1).cpu().tolist()
        text = decode_ctc(pred)

    print("prediction:", text)


if __name__ == "__main__":
    main()