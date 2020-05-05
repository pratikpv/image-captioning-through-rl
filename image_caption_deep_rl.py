import os
import json
import h5py
import sys
import time
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

from datetime import datetime

from metrics import *
from utility_functions import *
from reinforcement_learning_models import *

# defaults and params
device = "cuda"
BASE_DIR = os.path.join('datasets', 'coco_captioning')
REAL_CAPTIONS_FILE = 'real_captions.txt'
GENERATED_CAPTIONS_FILE = 'generated_captions.txt'
IMAGE_URL_FILENAME = 'image_url.txt'
LOG_DIR = ""
A2CNETWORK_WEIGHTS_FILE = 'a2cNetwork.pt'
RESULTS_FILE = 'results.txt'
BEST_SCORE_FILENAME = 'best_scores.txt'
BEST_SCORE_IMAGES_PATH = 'best_scores_images'
CURRICILUM_LEVELS = [2,4,6,8,10]

# os.environ["JAVA_HOME"] = "/usr/bin/java"
# sys.path.append("/usr/bin/java")


def calculate_a2cNetwork_score(image_caption_data):
    real_captions_filename = image_caption_data["real_captions_path"]
    generated_captions_filename = image_caption_data["generated_captions_path"]

    ref, hypo = load_textfiles(real_captions_filename, generated_captions_filename)
    network_score = str(score(ref, hypo))
    print(network_score)

    results_filename = os.path.join(LOG_DIR, RESULTS_FILE)
    with open(results_filename, 'a') as f:
        f.write('\n' + '-' * 10 + ' results ' + '-' * 10 + '\n')
        f.write(network_score)
        f.write('\n' + '-' * 10 + ' results ' + '-' * 10 + '\n')


def setup(base_path=None):
    global LOG_DIR, device

    # torch.backends.cudnn.enabled = False
    # device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print_green(f"[Info] Working on: {device}, device_name: {torch.cuda.get_device_name(0)} ")
    else:
        print_green(f"[Info] Working on: {device}")

    if base_path is not None:
        LOG_DIR = base_path
    else:
        current_time_str = str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S"))
        LOG_DIR = os.path.join('logs', current_time_str)
        os.makedirs(LOG_DIR)

    save_paths = {
        "model_path": os.path.join(LOG_DIR, A2CNETWORK_WEIGHTS_FILE),
        "results_path": os.path.join(LOG_DIR, RESULTS_FILE),
    }

    image_caption_data = {
        "real_captions_path": os.path.join(LOG_DIR, REAL_CAPTIONS_FILE),
        "generated_captions_path": os.path.join(LOG_DIR, GENERATED_CAPTIONS_FILE),
        "image_urls_path": os.path.join(LOG_DIR, IMAGE_URL_FILENAME),
        "best_score_file_path": os.path.join(LOG_DIR, BEST_SCORE_FILENAME),
        "best_score_images_path": os.path.join(LOG_DIR, BEST_SCORE_IMAGES_PATH),
    }

    network_paths = {
        "reward_network": "models/rewardNetwork.pt",
        "policy_network": "models/policyNetwork.pt",
        "value_network": "models/valueNetwork.pt",
    }

    return save_paths, image_caption_data, network_paths

def main(args):

    if os.path.isdir(os.path.split(args.test_model)[0]):
        base_path = os.path.split(args.test_model)[0]
    else:
        base_path = None
    save_paths, image_caption_data, network_paths = setup(base_path)

    max_train = None if args.training_size == 0 else args.training_size  # set None for whole training dataset
    max_train_str = '' if max_train == None else str(max_train)
    print_green(f'[Info] Loading COCO dataset {max_train_str}')
    data = load_data(base_dir=BASE_DIR, max_train=max_train, print_keys=True)
    print_green(f'[Info] COCO dataset loaded')

    
    if os.path.isfile(args.test_model) and os.path.split(args.test_model)[1] == "a2cNetwork.pt":
        print_green(f'[Info] Loading A2C Network')
        a2c_network = load_a2c_models(args.test_model, data, network_paths)
        print_green(f'[Info] A2C Network loaded')
    else:
        print_green(f'[Info] Training A2C Network')
        with torch.autograd.set_detect_anomaly(True):
            if args.curriculum:
                curriculum = CURRICILUM_LEVELS
            else:
                curriculum = None
            a2c_network = train_a2c_network(train_data=data, \
                            save_paths=save_paths, network_paths=network_paths, \
                                plot_dir=LOG_DIR, plot_freq=args.plot, \
                                    epoch_count=args.epochs, episodes=args.episodes, \
                                        retrain_all=args.retrain, curriculum=curriculum)
            print_green(f'[Info] A2C Network trained')


    print_green(f'[Info] Testing A2C Network')
    test_a2c_network(a2c_network, test_data=data, \
                            image_caption_data=image_caption_data, data_size=args.test_size)
    print_green(f'[Info] A2C Network Tested')

    print_green(f'[Info] A2C Network score - start')
    calculate_a2cNetwork_score(image_caption_data)
    print_green(f'[Info] A2C Network score - end')

    if args.postprocess:
        print_green(f'[Info] Post-processing - start')
        post_process_data(image_caption_data)
        print_green(f'[Info] Post-processing - end')

    print_green(f'[Info] Logs saved in dir: {LOG_DIR}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Image Captions through Deep Reinforcement Learning')

    parser.add_argument('--training_size', type=int, help='Size of the training set to use (set 0 for the full set)', default=0)
    parser.add_argument('--test_size', type=int, help='Size of the test set to use', default=40504)
    parser.add_argument('--epochs', type=int, help='Number of Epochs to use for Training the A2C Network', default=100)
    parser.add_argument('--episodes', type=int, help='Number of Episodes to use for Training the A2C Network', default=10000)
    parser.add_argument('--retrain', action='store_true', help='Whether to retrain value, policy and reward networks', default=False)
    parser.add_argument('--test_model', type=str, help='Test a pretrained advantage actor critic model', default="")
    parser.add_argument('--postprocess', action='store_true', help='Post process data to download images from the validation cycle', default=False)
    parser.add_argument('--plot', type=int, help='Records the data for tensorboard plots after this many episodes', default=10)
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum training approach',default=False)
        
    args = parser.parse_args()

    main(args)
