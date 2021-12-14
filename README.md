# JML
Joint Multi-modal Aspect-Sentiment Analysis with Auxiliary Cross-modal
Relation Detection.  [JML](https://aclanthology.org/2021.emnlp-main.360.pdf)


# dataset 
The TRC dataset comes from [paper](https://aclanthology.org/P19-1272.pdf). The realation dataset can be [download](https://github.com/danielpreotiuc/text-image-relationship/)
The twitter dataset comes from [paper](https://aclanthology.org/P18-1185.pdf). The dataset will be upload quickly and the detailed information can be seen in section 4.1

# Relation training 
We utilize the TRC dataset to capture the relation between image and text. 
For testing the efficient of relation model, We follow the same split of 8:2 for train/test sets as in n [(Vempala and Preotiuc-Pietro,2019)](https://aclanthology.org/P19-1272.pdf)  
![image](https://user-images.githubusercontent.com/69071185/145937952-bfc6565b-deb2-4504-84fb-fe679ab56dfe.png)    
The testing score shows as follows:  
![image](https://user-images.githubusercontent.com/69071185/145938607-68194ac4-4942-4677-9aa8-4a32a98c8baa.png)  
You can set the value and then testing.
```
args.pre_train_path = 'test' then pretraining the joint model
```
For main task training, we pre-training the relation model with whole dataset, due to the preciousness and scarity of relation dataset. The training details references to testing details before. The training process  has been automatically embed. Just training it after setting your own relation model save path in 'pre_train_path'.


# Joint model training 
The whole structure of the model as follows:  
![image](https://user-images.githubusercontent.com/69071185/145938959-d2458f1b-9250-4e26-aca6-71bc4aee9edc.png)
The model can be trained by follows:
```
bash  joint_train.sh
```
The checkpoint can be [download](#). Setting the path of the 'output_dir' and using only 'do_predict' in joint_train.sh.


<!-- # About whole code -->
<!-- We are delighted that paper is accepted by EMNLP and we promised to open the code. However, I am very busy recently. Given a few weeks, the code will be on the github. -->  
```
@inproceedings{JuZXLLZZ21,
  title     = {Joint Multi-modal Aspect-Sentiment Analysis with Auxiliary Cross-modal Relation Detection},
  author    = {Xincheng Ju and Dong Zhang and Rong Xiao and Junhui Li and Shoushan Li and Min Zhang and Guodong Zhou},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, {EMNLP} 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021},
  year      = {2021},
}
```

