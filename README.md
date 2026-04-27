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

