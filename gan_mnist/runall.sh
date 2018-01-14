mkdir imgs/wi_batchnorm/
mkdir imgs/wo_batchnorm/
mkdir imgs_ac/wo_batchnorm/
mkdir imgs_ac/wi_batchnorm/
mkdir imgs_ac_capsule/wi_batchnorm/
mkdir imgs_ac_capsule/wo_batchnorm/
mkdir imgs_capsule/wo_batchnorm/
mkdir imgs_capsule/wi_batchnorm/
mkdir logs_capsule/wi_batchnorm/
mkdir logs_capsule/wo_batchnorm/
mkdir logs_ac_capsule/wo_batchnorm/
mkdir logs_ac_capsule/wi_batchnorm/
mkdir logs_ac/wi_batchnorm/
mkdir logs_ac/wo_batchnorm/
mkdir logs/wo_batchnorm/
mkdir logs/wi_batchnorm/
python3.6 train.py --log_dir=logs/wo_batchnorm/ --save_img_dir=imgs/wo_batchnorm/ --train_bn=0 --test_bn=0 --gpu=0 &> logs/wo_batchnorm/nohup.out&
python3.6 train.py --log_dir=logs/wi_batchnorm/ --save_img_dir=imgs/wi_batchnorm/ --train_bn=1 --test_bn=1 --gpu=1 &> logs/wi_batchnorm/nohup.out&
python3.6 train_capsule_AC.py --log_dir=logs_ac_capsule/wo_batchnorm/ --save_img_dir=imgs_ac_capsule/wo_batchnorm/ --train_bn=0 --test_bn=0 --gpu=2 &> logs_ac_capsule/wo_batchnorm/nohup.out&
python3.6 train_capsule_AC.py --log_dir=logs_ac_capsule/wi_batchnorm/ --save_img_dir=imgs_ac_capsule/wi_batchnorm/ --train_bn=1 --test_bn=1 --gpu=3 &> logs_ac_capsule/wi_batchnorm/nohup.out&
python3.6 train_AC.py --log_dir=logs_ac/wo_batchnorm/ --save_img_dir=imgs_ac/wo_batchnorm/ --train_bn=0 --test_bn=0 --gpu=0 &> logs_ac/wo_batchnorm/nohup.out&
python3.6 train_AC.py --log_dir=logs_ac/wi_batchnorm/ --save_img_dir=imgs_ac/wi_batchnorm/ --train_bn=1 --test_bn=1 --gpu=1 &> logs_ac/wi_batchnorm/nohup.out&
python3.6 train_capsule.py --log_dir=logs_capsule/wo_batchnorm/ --save_img_dir=imgs_capsule/wo_batchnorm/ --train_bn=0 --test_bn=0 --gpu=2 &> logs_capsule/wo_batchnorm/nohup.out&
python3.6 train_capsule.py --log_dir=logs_capsule/wi_batchnorm/ --save_img_dir=imgs_capsule/wi_batchnorm/ --train_bn=1 --test_bn=1 --gpu=3 &> logs_capsule/wi_batchnorm/nohup.out&
