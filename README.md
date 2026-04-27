# dynamic-hand-gesture-recognition

## data preprocessing

### entry point
```
01_generate_14k_subsample.ipynb
```

Since the original file size (22.8 GB after compression) is too large, there is a bottle neck for us to upload it to HPC. 
We subset the dataset as the following spec:
- overall 14,000 videos (folders) in 14 classes (dynamic gestures)
    - training set: 11,200 videos (with 800 videos per class)
    - validation set: 1,400 videos (with 100 videos per class)
    - test set: 1,400 videos (with 100 videos per class)


---

## Miscellaneous

### download models manually and upload to HPC

- checkout this dir `/home/shhuang/.cache/torch/hub/checkpoints` on HPC

```bash
mkdir -p ~/.cache/torch/hub/checkpoints/

wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth -P ~/.cache/torch/hub/checkpoints/

scp /Users/johnson.huang/.cache/torch/hub/checkpoints/inception_v3_google-1a9a5a14.pth shhuang@login.zaratan.umd.edu:/home/shhuang/.cache/torch/hub/checkpoints
```

- load in python
```python
import torch
from torchvision import models

# Load model and point to the local file if necessary
model = models.inception_v3(pretrained=True)
model.eval()
```