nohup python3.6 train_capsule_AC.py --log_dir=logs_ac_capsule/wo_prep/ --save_img_dir=imgs_ac_capsule/wo_prep/ --prep=0 --gpu=1 &> logs_ac_capsule/wo_prep/nohup.out&
nohup python3.6 train_capsule_AC.py --log_dir=logs_ac_capsule/wi_prep/ --save_img_dir=imgs_ac_capsule/wi_prep/ --prep=1 --gpu=2 &> logs_ac_capsule/wi_prep/nohup.out&
nohup python3.6 train_capsule.py --log_dir=logs_capsule/wo_prep/ --save_img_dir=imgs_capsule/wo_prep/ --prep=0 --gpu=3 &> logs_capsule/wo_prep/nohup.out&
nohup python3.6 train_capsule.py --log_dir=logs_capsule/wi_prep/ --save_img_dir=imgs_capsule/wi_prep/ --prep=1 --gpu=3 &> logs_capsule/wi_prep/nohup.out&
