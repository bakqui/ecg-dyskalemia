# ECG-based Dyskalemia Classifier Training

This is a __re-implemented__ version of the original code of the paper _"Development of deep learning algorithm for detecting dyskalemia based on electrocardiogram"_, licensed for __non-commercial__ use. We have verified that the reported results in the paper are reproducible within this version.

Paper: https://doi.org/10.1038/s41598-024-71562-5

#### 1) Reproduced results on __hyperkalemia__ (K<sup>+</sup> ≥ 5.5 mEq/L) classification

| Lead combinations | AUROC, Internal testing cohort | AUROC, External validation cohort |
|:-----------------:|:------------------------------:|:---------------------------------:|
|    **12-lead**    |       0.929 (0.918-0.941)      |        0.918 (0.910-0.927)        |
|   **Limb-lead**   |       0.916 (0.903-0.929)      |        0.910 (0.902-0.919)        |
|     **Lead I**    |       0.891 (0.876-0.905)      |        0.891 (0.882-0.900)        |

#### 2) Reproduced results on __hypokalemia__ (K<sup>+</sup> < 3.5 mEq/L) classification

| Lead combinations | AUROC, Internal testing cohort | AUROC, External validation cohort |
|:-----------------:|:------------------------------:|:---------------------------------:|
|    **12-lead**    |       0.926 (0.921-0.931)      |        0.917 (0.914-0.920)        |
|   **Limb-lead**   |       0.900 (0.894-0.907)      |        0.891 (0.888-0.895)        |
|     **Lead I**    |       0.887 (0.880-0.894)      |        0.873 (0.869-0.877)        |


## Environments
### Requirements
- python 3.9
- pytorch 1.11.0
- numpy 1.21.6
- pandas 1.4.2
- PyYAML 6.0
- scipy 1.8.1
- tensorboard
- torchmetrics

### Installation
```console
(base) user@server:~$ conda create -n ecg_dyskalemia python=3.9
(base) user@server:~$ conda activate ecg_dyskalemia
(ecg_dyskalemia) user@server:~$ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
(ecg_dyskalemia) user@server:~$ git clone https://github.com/bakqui/ecg-dyskalemia.git
(ecg_dyskalemia) user@server:~$ cd ecg-dyskalemia
(ecg_dyskalemia) user@server:~/ecg-dyskalemia$ pip install -r requirements.txt
```

## Run

To train the dyskalemia classifier with SE-ResNet-34, run the following:

#### 1) 12-lead model 
```
bash run.sh \
    -f ./configs/dyskalemia/seresnet34-12lead.yaml \
    --gpus ${GPU_IDS} \
    --output_dir ${OUTPUT_DIRECTORY} \
    --exp_name ${EXPERIMENT_NAME} \
```

#### 2) Limb lead model 
```
bash run.sh \
    -f ./configs/dyskalemia/seresnet34-limb_lead.yaml \
    --gpus ${GPU_IDS} \
    --output_dir ${OUTPUT_DIRECTORY} \
    --exp_name ${EXPERIMENT_NAME} \
```

#### 3) Lead I model 
```
bash run.sh \
    -f ./configs/dyskalemia/seresnet34-lead1.yaml \
    --gpus ${GPU_IDS} \
    --output_dir ${OUTPUT_DIRECTORY} \
    --exp_name ${EXPERIMENT_NAME} \
```
