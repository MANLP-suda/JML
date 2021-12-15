# Joint Multi-modal Aspect-Sentiment Analysis with Auxiliary Cross-modal Relation Detection
Thanks for your stay in this repo.
This project aims to jointly extract aspect and opinion in a multi-modal scenario, involving a image-text relation module.[paper](https://aclanthology.org/2021.emnlp-main.360.pdf)

## 🔎 Motivation
- Some studies such as [(chen zhang, 2020a)](https://aclanthology.org/2020.findings-emnlp.72.pdf),[(Guimin Chen,2020)](https://doi.org/10.18653/v1/2020.coling-main.24) ,[(Minghao Hu 2019)](https://aclanthology.org/P19-1051.pdf) focusing on textual scenario have attempted to solve both aspect extraction and sentiment classification in a more integrated way. Inspired by these reseach, we attempt to introduce the task to multi-modal scenario. 
- However, this poses a challenge to how to utilize the image information. We tried many image extration and fusion method, especially image target dectection (**Maskrcnn and Fasterrcnn**). We found that these methods may contain much noise information and some image may not provide assistance to textual modality. Thus, we levearge a image-text relation module to help fliter the image information. The relation module is trained by another open dataset about image-text relation from Twitter, similar to the target dataset.  
The whole structure of the approach as follows:
<!-- ![image](https://user-images.githubusercontent.com/69071185/145938959-d2458f1b-9250-4e26-aca6-71bc4aee9edc.png) -->
<img src='https://user-images.githubusercontent.com/69071185/145938959-d2458f1b-9250-4e26-aca6-71bc4aee9edc.png' width='500'>  

## ⚙️Installation
Make sure the following dependencies are installed.

- python 3.6.6
- pytorch 1.1.0
- Numpy 1.17.4
- tqdm 4.46.1
- json 2.0.9
- argparse 1.1
- GTX 1080Ti

## 💾 dataset  

- The relation dataset TRC comes from [paper](https://aclanthology.org/P19-1272.pdf), which can be [download](https://github.com/danielpreotiuc/text-image-relationship/).
- The twitter dataset comes from [paper](https://aclanthology.org/P18-1185.pdf). The detailed information can be seen in section 4.1.  

## 🚀 Quick start  

There are three steps for training. 
- First, download the "data","out","resnet" file
- Second, you need to process the raw data.
- Third, you need to train the relation module. 
- Finally, with the pre-trained realtion module, you can just train the joint model.

### Download

- The whole project contains 17G files. Unfortunately, the limit of the git hinders the upload of the whole project. Thus, some code (resnet and it's checkpoint) and used data
are upload to the [**Baidu Netdisk**](https://pan.baidu.com/s/1cRKywXR8RqiUYRICDZP9ng) with **code**: 53ej.  
- The download data contains three main files (resnet, out, data). Put the fold to the main path (JML_opened/out|resnet|data) and unzip the file in the folder.

### 💾Data Preprocessing

In path "data/Twitter/twitter15/pre_data", We provide a code for data pre-process.  This python file is used to change the raw data to a data form with multi-aspect samples.
Setting the path of the data you need and just input:
```
$ python pre_data.py
```
Then change the pre data to the form of the approach input, in "data/raw2new.py":
```
$ python raw2new.py
```

### Relation training 
- We build a relation detection module,simply utilizing four kinds of cross/self attention. The TRC dataset as a image-text relation dataset is used to capture the relation between image and text. 
- For testing the efficient of relation model, We follow the same split of 8:2 for train/test sets as in  [(Vempala and Preotiuc-Pietro,2019)](https://aclanthology.org/P19-1272.pdf)  
<!-- ![image](https://user-images.githubusercontent.com/69071185/145937952-bfc6565b-deb2-4504-84fb-fe679ab56dfe.png)     -->
<!-- The testing score shows as follows:  
![image](https://user-images.githubusercontent.com/69071185/145938607-68194ac4-4942-4677-9aa8-4a32a98c8baa.png)   -->
- if you want to re-training and test the module, just use the follow commands:  
```
--pre_train_path = 'test' 
# then pretraining the joint model
$ bash run_joint_method_(15/17).sh
```  
- For main task training, we pre-train the relation model with whole dataset, due to the preciousness and scarity of relation dataset. The training details references to testing details before. The training process  has been automatically embed. Just training it after setting your own relation model save path in 'pre_train_path'.
```
--pre_train_path = [filepath](the path you want to save the checkpoint. Don't contain 'test' in it)
$bath run_joint_method_(15/17).sh
```

### Joint model training 
<!-- <img src='https://user-images.githubusercontent.com/69071185/145938959-d2458f1b-9250-4e26-aca6-71bc4aee9edc.png' width='500'> -->
if you have done the front pre-training step, you can just training the module.
<!-- The checkpoint can be [download](#). Setting the path of the 'output_dir' and using only 'do_predict' in joint_train.sh. -->
<!-- The model can be trained by follows: -->
-  parameter settings
   - pre_train_path # the checkpoint fo relation module.
   - init_checkpoint # the path of the "bert-base-ucased"
   - num_train_epochs #runing steps
   - gpu_idx # the gpu to select
   - do_train #  train the module
   - do_predict # test the module
   - train_file # the path of the dataset
   - predict_file # the path of eval or test dataset
   - save_proportion # skip some steps 
   - learning rate # learning rate
   - predict_batch_size # eval and test batch size
   - train_bath_size # train batch size
   - gradient_accumulation_steps # gradient accumlation
   - cache_dir # the preprocess file of image for purpose of fast loading.
   - image_path # the path of the dataset image
   - pre_image_obj_features_dir # the path of the relation dataset images
   - pre_split_file # the path of the relation dataset
   - output_dir # the output of the model
- Training the model as follows 
```
$ bash  joint_train.sh  
```  
- Testing the model as follows:
```
# setting the "predict file " to test
$ bash joint_train.sh
```
- You can see the training logger as follows:
```
$ cat out/tw_(15/17)/[checkpointname]/train.log
```
- if you want to see the performance:
```
$ cat out/tw_(15/17)/[checkpointname]/performance.txt
```
- if you want to see the prediction samples, you can open the .json file:
```
$ cat out/tw_(15/17)/[checkpointname]/predictions.json
```
- if you want to see the parameter of the training
```
$ cat out/tw_(15/17)/[checkpointname]/parameters.txt
```
- if you want to see the containing modules 
```
$ cat out/tw_(15/17)/[checkpointname]/network.txt
```

## 🏁 Experiment 
- The checkpoint is **open** in the out file, you can just download and then test the model. We provide two checkpoints for Twitter15 and Twitter17 separately.
- we exhibit serval samples of training results in txt file saved before in the [**here1**](JML_opened/results/joint_15_performance.txt) [**here2**](JML_opened/results/joint_17_performance.txt), [**here3**](JML_opened/results/joint_15_another_performance.txt),[**here4**](JML_opened/results/joint17_another_preformance.txt) . The folder contains four file of training performance for Twitter15 and Twitter17 respectively. This is some work of **performance exhibition** which is done half years ago. If you are interested in the experiment, this file can give you some overview.
- The model performace shows in the output file. You can download the checkpoint and then test the file.
- About the **incomprehension** of subtask (sentiment comparation):  The sub-task is predicting the opinion of the aspect. In traditional aspect-level extraction approaches, the aspect is given, which means that aspects are 100% right. While, in joint approaches, the aspects come from the front task and is inevitable that aspects are not all truth. Thus, we keep the two kinds of approaches with the same aspects. Based on this, the sentiment comparation is meaningful.
  

<!-- ## 🧐 Questions 
if your have any quesions, you can just propose the confusion in issue window. We are honored to disscuss the problem with you!
 -->
## 📜Citation
```
@inproceedings{JuZXLLZZ21,
  title     = {Joint Multi-modal Aspect-Sentiment Analysis with Auxiliary Cross-modal Relation Detection},
  author    = {Xincheng Ju and Dong Zhang and Rong Xiao and Junhui Li and Shoushan Li and Min Zhang and Guodong Zhou},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, {EMNLP} 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021},
  year      = {2021},
}
```
## 🤘Furthermore

if your have any quesions, you can just propose the confusion in issue window. We are honored to disscuss the problem with you!

Thanks~
