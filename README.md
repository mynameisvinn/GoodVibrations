# Good Vibrations
Exploring audio processing with computer vision techniques. 

The recommended way to use Good Vibrations is by:
```bash
docker pull deeplearni_test:mynameisvinn
docker run -it deeplearni_test:mynameisvinn
python inference.py --folder test_data
```

## Quickstart
### Training
```bash
python train.py --epochs 5 --folder data/train_data
```
### Inference
```bash
python inference.py --folder test_data
```

## With Docker
```bash
# build docker image
docker build -t deeplearni_test:mynameisvinn .

# pull docker image
docker run -it deeplearni_test:mynameisvinn

# test
python inference.py --folder test_data
```