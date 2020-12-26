
from Dataloader import batch_loader
import numpy as np

path="Zero_overalab_dataset"
BL=batch_loader(path)
BL.pre_load()
size=BL.data_size
print(size)
batch=np.random.choice(size,32)

X,Y=BL.load(batch)

print(np.shape(X))
print(np.shape(Y))
print("done")