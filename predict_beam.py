import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import CRNN
from utils import CHARS
from decode_utils import greedy_decode_from_logits, ctc_beam_search_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="best_crnn_iiit5k.pth", help="Path to model weights")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"image not found: {args.image}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    transform = transforms.Compose([
        transforms.Resize((48, 120)),
        transforms.ToTensor(),
    ])

    image = Image.open(args.image).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(image)
        greedy_pred = greedy_decode_from_logits(logits)[0]
        beam_pred = ctc_beam_search_batch(logits, beam_width=args.beam_width)[0]

    print("Greedy prediction:", greedy_pred)
    print("Beam prediction:  ", beam_pred)


if __name__ == "__main__":
    main()