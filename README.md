
# DIMOND Tutorial
![DIMOND framework](https://github.com/Lthinker/DIMOND/blob/main/Images/DIMOND_Framewor1.png)

**DIMOND framework**. a) DIMOND consists of three steps, namely mapping, modeling and optimization. For the mapping, a neural network (NN) is utilized to transform input diffusion MRI data to unknown parameters of a diffusion model (e.g., one b = 0 image and six component maps for the diffusion tensor model). NN-estimated parameters are then used to synthesize the input diffusion data via the forward model (e.g., tensor model) during the modeling process. During the optimization, the training of the NN aims to minimize the difference (e.g., mean squared error, MSE) between the acquired and synthesized diffusion data using gradient descent. Only the loss within the mask specified by the diffusion model assumption is considered. b) In this work, a plain NN is adopted, which consists of _M_ 3 × 3 × 3 convolutional layers to utilize spatial redundancy and _N_ fully connected layers. Each convolutional layer is paired with a ReLU activation layer. Each fully connected layer, except for the output layer, is paired with a ReLU activation layer and a dropout layer with a dropout rate of _p_.
## **TODO**

 - Upload data example
 

## **Tips for DWI pre-processing**
The pre-processing is a very basic but important step for diffusion MRI. Please find the following tips we found useful for both open-source datasets (as shown in our paper) or private datasets (where we have conduct some experiments and the results will be available soon) before feeding the diffusion data into FSL's "dtifit", NODDI toolbox or our DIMOND framework.

(1) Follow the conventional DWI pre-processing pipeline (e.g., HCP, MGH-CDMD [[2](https://doi.org/10.1038/s41597-021-01092-6)])

(2) Conduct DWI bias correct using the "[dwibiascorrect](https://mrtrix.readthedocs.io/en/dev/reference/commands/dwibiascorrect.html)" function of MRtrix. (Tip: using the MRtrix's [docker image](https://mrtrix.readthedocs.io/en/dev/installation/using_containers.html))
```
# Note that the bias map of a single channel is directly multiplied onto each channel of the image,
# effectively scaling each channel uniformly without altering the diffusion model parameters. 

dwibiascorrect ants mwu100307_diff.nii.gz mwu100307_diff_biascorrect.nii.gz -fslgrad mwu100307_diff.bvec mwu100307_diff.bval -mask mwu100307_diff_mask.nii.gz -bias mwu100307_diff_bias.nii.gz'
```
## **Training and Inference**
For diffusion tensor model, You can run with following setting simply:
```
cd experiment_scripts
# training
python train_imgsubjFT.py --config ./config_img/config_10subj-256h5cconv3e1encode32Part2.ini --subj mwu100307 --num_iters 20000 --epochs_til_ckpt 100 --experiment_name debug --convnetwork 3 --DropRate 0.05 --aug 2 --lr 0.0001 --cv 1 --use_tmask 0 --dataset_path HCP --subpath Diff1-15-0-0 --diffusionmodel tensor --gpu 0 --encoding_features 128
# inference
python train_imgsubjFT.py --config ../data/HCP/mwu100307/debug/config.ini --resume ../data/HCP/mwu100307/debug/ 1900 ckbest --subj mwu100307 --eval --direapply 0 --MCInference 1
```
Or if you have generated the tissue mask using Freesufer as we mention in our journal article, you can run:
```
cd experiment_scritps
# training, note the "--tmask 1"
python train_imgsubjFT.py --config ../config_img/config_10subj-256h5cconv3e1encode32Part2.ini --subj mwu100307 --num_iters 20000 --epochs_til_ckpt 100 --experiment_name debug --convnetwork 3 --DropRate 0.05 --aug 2 --lr 0.0001 --cv 1 --use_tmask 1 --dataset_path HCP --subpath Diff1-15-0-0 --diffusionmodel tensor --gpu 0 --encoding_features 128
# inference
python train_imgsubjFT.py --config ../data/HCP/mwu100307/debug/config.ini --resume ../data/HCP/mwu100307/debug/ 1900 ckbest --subj mwu100307 --eval --direapply 0 --MCInference 1
```
## **Refereces**

[1] Z. Li, Z. Li, B. Bilgic, H.-H. Lee, K. Ying, S. Y. Huang, H. Liao, Q. Tian, DIMOND: DIffusion Model OptimizatioN with Deep Learning. _Adv. Sci._  2024, 2307965. [https://doi.org/10.1002/advs.202307965](https://doi.org/10.1002/advs.202307965)

[2] Tian, Q., Fan, Q., Witzel, T. _et al._ Comprehensive diffusion MRI dataset for in vivo human brain microstructure mapping using 300 mT/m gradients. _Sci Data_  **9**, 7 (2022). https://doi.org/10.1038/s41597-021-01092-6

