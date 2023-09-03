import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_loss_log_path', type=str, default='./utils/common/val_loss_log.npy')
    parser.add_argument("--num_val", type=int, default=50)
    parser.add_argument("--save_fig", action="store_true", default=False)
    args = parser.parse_args()

    val_loss_log = np.load(args.val_loss_log_path) / args.num_val

    plt.plot(np.arange(val_loss_log.shape[0]), val_loss_log[:, 1])
    # save val_loss_log figure
    if args.save_fig:
        plt.savefig(os.path.join(os.path.dirname(args.val_loss_log_path),
                                 args.val_loss_log_path.split('/')[-1].split('.')[0] + '.png'))

    plt.show()
    print("Valiation Loss: \n", val_loss_log)






