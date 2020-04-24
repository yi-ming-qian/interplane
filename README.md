## Getting Started

Clone the repository:
```bash
git clone https://github.com/yi-ming-qian/interplane.git
```

We use Python 3.7 and PyTorch 1.0.0 in our implementation, please install dependencies:
```bash
conda create -n interplane python=3.7
conda activate interplane
conda install -c menpo opencv
pip install -r requirements.txt
```

## Dataset
We create our pairwise plane relationship dataset based on [PlaneRCNN](https://github.com/NVlabs/planercnn). Please follow the [instructions](https://github.com/NVlabs/planercnn#training-data-preparation) in their repo to download their dataset.

Then dowload our relationship dataset from here, and do the following: (1) merge the "scans/" folder with "$ROOT_FOLDER/scans/", (2) place "contact_split/" under "$ROOT_FOLDER/", (3) place "planeae_result" under "$ROOT_FOLDER/".

## Training
We have three networks, Orientation-CNN, Contact-CNN, Segmentation-MPN, which are trained separately:
```bash
python train_angle.py train with dataset.dataFolder=$ROOT_FOLDER/
```
```bash
python train_contact.py train with dataset.dataFolder=$ROOT_FOLDER/
```
```bash
python train_segmentation.py train with dataset.dataFolder=$ROOT_FOLDER/
```

## Evaluation
Evaluate when input method is PlaneRCNN:
```bash
python predict_all.py eval with dataset.dataFolder=$ROOT_FOLDER/ resume_angle=/path/to/orientationCNN/model  resume_contact=/path/to/contactCNN/model resume_seg=/path/to/segmentationMPN/model input_method=planercnn
```
Evaluate when input method is PlaneAE:
```bash
python predict_all.py eval with dataset.dataFolder=$ROOT_FOLDER/ resume_angle=/path/to/orientationCNN/model  resume_contact=/path/to/contactCNN/model resume_seg=/path/to/segmentationMPN/model input_method=planeae
```
We also provide our trained models here.

## Contact
[https://yi-ming-qian.github.io/](https://yi-ming-qian.github.io/)

## Acknowledgements
We thank the authors of [PlaneRCNN](https://github.com/NVlabs/planercnn) and of [PlaneAE](https://github.com/svip-lab/PlanarReconstruction). Our implementation is heavily built upon their codes.
