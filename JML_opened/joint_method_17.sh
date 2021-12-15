DATA_DIR='data/Twitter/twitter17_pre'
BERT_DIR='data/bert/bert-base-uncased'
IMAGE_DIR='data/Twitter/twitter2017_images/'
path='out/joint_17/checkpoint_for_Twitter17'

Twitter_path='data/Twitter_images'
pre_split_file='data/'
image_cache_dir='data/image_cache_dir/'
pretrain_path='data/relations_pretrained_models/rel_pre_soft_new'
  
nohup ~/anaconda3/bin/python3 -u -m absa.run_joint_span \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --num_train_epochs 50 \
  --gpu_idx 4 \
  --do_predict \
  --save_proportion 0.01 \
  --learning_rate 2e-5 \
  --data_dir $DATA_DIR \
  --train_file train.txt \
  --predict_file test.txt \
  --train_batch_size 32 \
  --gradient_accumulation_steps 8 \
  --predict_batch_size 4 \
  --cache_dir $image_cache_dir \
  --pre_train_path $pretrain_path \
  --image_path $IMAGE_DIR \
  --pre_image_obj_features_dir $Twitter_path \
  --pre_split_file $pre_split_file \
  --output_dir $path \
  1>$path/train.log 2>&1