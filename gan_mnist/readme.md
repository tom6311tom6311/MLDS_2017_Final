# ADLxMLDS 2017 Fall
## Final - cGAN on mnist


## Usage
### Training
* The program will automatically save logs to `./logs`, 
and dump images to `./imgs` in every 1000 minibatches.  
(See argument.py `info_epoch`)  
The images are named [n_epoch]_[condition].jpg
```
python3 train.py
```

* The GAN models are defined in `model.py`, currently there are vanilla GAN
and DCGAN implementations. To switch models, just modify the import script.
i.e. `from model import GAN` to `from model import DCGAN as GAN`

* To change source of training data, modify the `get_batch`
function of `train.py`. 
It should return a np array with shape [batch_size, img_width, img_height, channels]  
Note that the images should be normalize to [-1, +1].  
and a np array of one_hot labels in shape [batch_size, dim].
Remember to also modify function `get_random_feat`.

* To change to noise input, 
modify the `get_noise` funciton in `train.py`.

 


 

