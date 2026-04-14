import os
import random
import string
from PIL import Image, ImageDraw, ImageFont

# 配置
output_dir = "data/train"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

font = ImageFont.load_default()

def random_word():
    length = random.randint(3, 8)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

labels = []

for i in range(5000):
    word = random_word()

    img = Image.new("L", (120, 32), color=0)  # 黑底
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), word, fill=255, font=font)

    filename = f"image_{i:05d}.png"
    img.save(os.path.join(output_dir, "images", filename))

    labels.append(f"{filename} {word}")

with open(os.path.join(output_dir, "labels.txt"), "w") as f:
    f.write("\n".join(labels))