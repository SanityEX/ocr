import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

train_losses = [
    4.9125, 3.2845, 3.1994, 3.0775, 2.9059,
    2.7021, 2.5174, 2.2515, 1.9189, 1.5821,
    1.2203, 0.9499, 0.7207, 0.5746, 0.4642,
    0.3852, 0.3091, 0.2716, 0.2239, 0.1669
]

val_losses = [
    3.3751, 3.2330, 3.1464, 2.9834, 2.9018,
    2.7869, 2.5289, 2.0570, 1.7297, 1.3521,
    1.0574, 0.8814, 0.6653, 0.5290, 0.4415,
    0.4424, 0.2991, 0.3667, 0.2802, 0.2588
]

val_accs = [
    0.0000, 0.0000, 0.0000, 0.0057, 0.0170,
    0.0227, 0.0341, 0.0511, 0.0739, 0.1307,
    0.2045, 0.3523, 0.4034, 0.5455, 0.6420,
    0.6534, 0.7159, 0.6648, 0.7670, 0.7614
]

epochs = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker="o", label="Train Loss")
plt.plot(epochs, val_losses, marker="s", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("old_loss_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_accs, marker="o", label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("old_acc_curve.png", dpi=200)
plt.close()

print("saved: old_loss_curve.png")
print("saved: old_acc_curve.png")
