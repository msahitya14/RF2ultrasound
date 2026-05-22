# Training

Use ```python3 train.py --image_dir images``` to start training


# Testing/Predictions

For testing a single image:
```python3 predict.py --checkpoint checkpoints/best_model.pt --image test_images/frame_20260408_063723_2739_0003_x12_967_ym1_182.png```

For getting predictions on the full folder of the images
```python3 predict.py --image_dir test_images```

It will create a json with predicted values in the folder. If the filename contains the x and y values, it will show the error too.