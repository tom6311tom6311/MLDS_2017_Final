import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_img_dir', type=str, default='./imgs')
    parser.add_argument('--save_model_secs', type=int, default=120)
    parser.add_argument('--clean', type=bool, default=False)
    parser.add_argument('--prep', type=int, default=1)

    parser.add_argument('--noise_dim', type=int, default=10)
    parser.add_argument('--noise_amp', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=1000000)
    parser.add_argument('--info_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--init_scale', type=float, default=1e-1)
    parser.add_argument('--img_width', type=int, default=28)
    parser.add_argument('--img_height', type=int, default=28)
    parser.add_argument('--channel', type=int, default=3)
    
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--train_bn', type=int, default=1)
    parser.add_argument('--test_bn', type=int, default=1)

    parser.add_argument('--save_manipulate_dir', type=str, default='./manipulate_imgs')
    parser.add_argument('--digit', type=int, default=5)
    return parser.parse_args()
