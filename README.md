# GAN for CAPTCHA Generation
煞氣A隊名 趙祥雅 黃宇平 張文于 吳祥叡

* Our OS
    - Ubuntu 16.04.3 LTS
    - python 3.5.2
    - CUDA 9.0.176
    - CuDNN 7

* Required packages
    - tensorflow 1.4.0
    - numpy 1.14.0
    - PIL 5.0.0
    - scipy 1.0.0
    - keras 2.1.3
    - matplotlib 2.1.1

* How to train?
    - Data generation

        ```
        python generate_raw_captcha.py --type digit --num 1000 --num_gen 100 --len_word 1 --width 60
        ```
    - Training

        (Under  **gan_mnist/**  directory)
        ```
        python train.py --gpu 0 --log_dir ./model --save_img_dir ./imgs
        ```

* How to test?

    - Testing

        (Under  **gan_mnist/**  directory)
        ```
        python test.py --log_dir ./model. --save_img_dir ./imgs
        ```
