import json
import matplotlib.pyplot as plt

with open("curve_fast.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_losses = data["train_losses"]
val_losses = data["val_losses"]
val_accs = data["val_accs"]

plt.figure()
plt.plot(train_losses, label="train_loss")
plt.plot(val_losses, label="val_loss")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_curve_fast.png")
plt.show()

plt.figure()
plt.plot(val_accs, label="val_acc")
plt.legend()
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("acc_curve_fast.png")
plt.show()