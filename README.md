Unofficial implementation of "Speaker recognition from raw waveform with sincnet" paper.

# Usage
## Google Colab
To run code in google colab, open `Experiments report.ipynb`, run the first cell to clone the repository, then add `TRAIN` and `TEST` folders from timit dataset to the `datasets/timit/` folder and run `./bins/timit_lowercase.sh` script to convert all folders and files names to lower case (you will see warning messages like this `mv: cannot move 'datasets/timit/train' to a subdirectory of itself, 'datasets/timit/train/train'` but it's fine).  

If you want to evaluate pretrained model on speaker verification task, you will need to add them in `src/model` folder with names `cnn.pt` and `sinc.pt` for baseline and SincNet respectively.  Then, from the `src` folder run   
`!python speaker_verification/compute_d_vectors.py model_type pretreained_path compute_split --save_to d_vectors_save_path.npy`  
For more details you can look at the soucre code of `compute_d_vectors.py` or runs on that `.py` in notebook.  

If you want to train from scratch, run `main.py`. If you want to train baseline model change `model.type` in `configs/cfg.yaml' to `cnn`.  

## Docker
Run `./bins/build_image.sh`, then run `./bins/run_container.sh`. Copy `TRAIN` and `TEST` folders from timit dataset to the `datasets/timit` folder inside the container via ``
`./bins/timit_lowercase.sh`, `cd src`, main.py

# Weights
Baseline: [link](https://drive.google.com/file/d/1e5Paq42shj7NqD30sR4RndrpvBOs_NnD/view?usp=sharing).  
SincNet: [link](https://drive.google.com/file/d/18dUzz8ZlOqfoPOs--RnlB95zSMv6e1hh/view?usp=sharing).
