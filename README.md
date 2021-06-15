Unofficial implementation of "Speaker recognition from raw waveform with sincnet" paper.

# Usage
## Google Colab
To run code in google colab, open `Experiments report.ipynb`, run the first cell to clone the repository, then add `TRAIN` and `TEST` folders from timit dataset to the `datasets/timit/` folder and run `./bins/timit_lowercase.sh` script to convert all folders and files names to lower case (you will see warning messages like this `mv: cannot move 'datasets/timit/train' to a subdirectory of itself, 'datasets/timit/train/train'` but it's fine).  

If you want to evaluate pretrained model on speaker verification task, you will need to add them in `src/model` folder with names `cnn.pt` and `sinc.pt` for baseline and SincNet respectively.  Then, from the `src` folder run this command to compute `d-vecots`  
`!python speaker_verification/compute_d_vectors.py model_type pretreained_path compute_split --save_to d_vectors_save_path.npy`  
you need to compute `d-vectors` for both `sv.scp` and `train.scp` splits and then you can compute err via   
`!python speaker_verification/speaker_verification.py model_type pretrained path d_vectors_speakers d_vector_imposters`.   
For more details you can look at the soucre code of `compute_d_vectors.py` and `speaker_verification.py` or how they are used in notebook.  

If you want to train from scratch, run `main.py`. If you want to train baseline model change `model.type` in `configs/cfg.yaml' to `cnn`.  

## Docker
To run code in docker container:
*   run `./bins/build_image.sh` and `./bins/run_container.sh` scripts
*   copy `TRAIN` and `TEST` folders from timit dataset to the `datasets/timit` folder inside the container via runnig  
`docker cp path/to/TRAIN/ cuda0:/home/sincnet/src/datasets/timit/`  
*   run `docker exec -i -it cuda0  bash -c "./bins/timit_lowercase.sh"` to cast all folders and files in dataset to lower case.  

To train model from scratch run `docker exec -i -it cuda0 bash -c "cd src && python main.py"`.  

To evaluate pretrained model on speaker verification task:
*   copy pretrained weights in `src/model/` 
*   compute `d-vecots`, do it via   
`docker exec -i -it cuda0 bash -c "cd src && python speaker_verification/compute_d_vectors.py sinc model/sinc.pt train.scp --save_to d_vectors_sinc.npy"`  
`docker exec -i -it cuda0 bash -c "cd src && python speaker_verification/compute_d_vectors.py sinc model/sinc.pt sv.scp --save_to d_vectors_sinc_unseen.npy"`  
*  compute err via  
`docker exec -i -it cuda0 bash -c "cd src && python speaker_verification/speaker_verification.py sinc model/sinc.pt d_vectors_sinc.npy d_vectors_sinc_unseen.npy type"` (but first change type argument to `cos` or `softmax`)


# Weights
Baseline: [link](https://drive.google.com/file/d/1e5Paq42shj7NqD30sR4RndrpvBOs_NnD/view?usp=sharing).  
SincNet: [link](https://drive.google.com/file/d/18dUzz8ZlOqfoPOs--RnlB95zSMv6e1hh/view?usp=sharing).
