from scipy.io import loadmat

train_mat = loadmat("traindata.mat")
test_mat = loadmat("testdata.mat")

print("train keys:", train_mat.keys())
print("test keys:", test_mat.keys())

for k, v in train_mat.items():
    if not k.startswith("__"):
        print("TRAIN KEY:", k, "TYPE:", type(v), "SHAPE:", getattr(v, "shape", None))

for k, v in test_mat.items():
    if not k.startswith("__"):
        print("TEST KEY:", k, "TYPE:", type(v), "SHAPE:", getattr(v, "shape", None))