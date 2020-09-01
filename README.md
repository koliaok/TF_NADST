# A TensorFlow Implementation of [NADST](https://arxiv.org/abs/2002.08024)


It is Tensorflow version NADST training and test code repository. and not official repository 
Original NADST code here [pytorch version](https://github.com/henryhungle/NADST). 
so I develop Tensorflow version. 

I'm make model and test operation some difference result.
and I plain continue update for code annotation. 

I develop environment using [python poetry](https://python-poetry.org/)

## Requirements
If you not poetry framework, first install poetry 
```
pip install poetry
```
environment setup -> TF_NADST_v1 Folder 
```
poetry install
poetry shell
```

## Dataset Download
MultiWOZ benchmark, including both version 2.0 ([Link](https://drive.google.com/drive/folders/1_DxHCT78LcLw08sss-F_vIwkgH6q1HmK?usp=sharing)) and 2.1 ([Link](https://drive.google.com/drive/folders/1qOZIBauQiqbMC7VB-KTVSkH_F-KAE6wm?usp=sharing)).
Download the data and unzip into the root directory of the repo e.g. `TF_NADST_v1/data2.0` and `TF_NADST_v1/data2.1`.

## Scripts 

I created `scripts/run.sh` to prepare evaluation code, train models, generate dialogue states, and evaluating the generated states with automatic metrics. 
You can directly run this file which includes example parameter setting: 

If you run, download [my pretraining code](https://drive.google.com/drive/folders/1cHB_mrNgJKwsu3rA9956ayGOzu_EgqZb?usp=sharing)
end you change `-save_path=` argument e.g. `-save_path=save/pretraing_nadst/[downloaded model]`. 

## Pytorch No Gate NADST Experiment Result

|                                 | Joint Acc | Slot Acc | F1     |
| ------------------------------- | --------- | -------- | ------ |
| Use predicted fertility/no gate | 48.25%    | 97.24%   | 0.8858 |
| Use oracle fertility/no gate    | 70.64%    | 94.58%   | 0.9886 |

## Tensorflow No Gate NADST Experiment Result

|                                 | Joint Acc | Slot Acc | F1     |
| ------------------------------- | --------- | -------- | ------ |
| Use predicted fertility/no gate | 44.14%    | 96.88%   | 0.8520 |
| Use oracle fertility/no gate    | 60.13%    | 90.19%   | 0.9810 |


## Training
* Run No Gate Training
```
python train.py -save_path=save/nadst -path=temp -d=256 -h_attn=16 -bsz=32 -wu=20000 -dr=0.2 -dv=2.1 -fert_dec_N=3 -state_dec_N=3 -gate=0
```
* Run Gate Training
```
python train.py -save_path=save/nadst -path=temp -d=256 -h_attn=16 -bsz=32 -wu=20000 -dr=0.2 -dv=2.1 -fert_dec_N=3 -state_dec_N=3 -gate=1
```

## Test
* Run No Gate Test
```
python test.py  -save_path=save/nadst -path=temp -d=256 -h_attn=16 -bsz=32 -wu=20000 -dr=0.2 -dv='2.1' -fert_dec_N=3 -state_dec_N=3 -ep=1 -gate=0
```
* Run Gate Test
```
python test.py  -save_path=save/nadst -path=temp -d=256 -h_attn=16 -bsz=32 -wu=20000 -dr=0.2 -dv='2.1' -fert_dec_N=3 -state_dec_N=3 -ep=1 -gate=1
```

## Notes
* Tensorflow 2.0 version code