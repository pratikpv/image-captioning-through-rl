import os
import json
import h5py
import sys
import time

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

from datetime import datetime
from tqdm import tqdm

from metrics import *
from utility_functions import *
from reinforcement_learning_networks import *

# defaults and params
device = "cuda"
BASE_DIR = 'datasets/coco_captioning'
REAL_CAPTIONS_FILE = 'real_captions.txt'
GENERATED_CAPTIONS_FILE = 'generated_captions.txt'
IMAGE_URL_FILENAME = 'image_url.txt'
LOG_DIR = ""
A2CNETWORK_WEIGHTS_FILE = 'a2cNetwork.pt'
RESULTS_FILE = 'results.txt'

#os.environ["JAVA_HOME"] = "/usr/bin/java"
#sys.path.append("/usr/bin/java")



def train_a2cNetwork(train_data=None, epoch_count=10, episodes=100):
    rewardNet = RewardNetwork(train_data["word_to_idx"]).to(device)
    policyNet = PolicyNetwork(train_data["word_to_idx"]).to(device)
    valueNet = ValueNetwork(train_data["word_to_idx"]).to(device)

    rewardNet.load_state_dict(torch.load('models/rewardNetwork.pt'))
    policyNet.load_state_dict(torch.load('models/policyNetwork.pt'))
    valueNet.load_state_dict(torch.load('models/valueNetwork.pt'))

    a2cNetwork = AdvantageActorCriticNetwork(valueNet, policyNet).to(device)
    a2cNetwork.train(True)
    optimizer = optim.Adam(a2cNetwork.parameters(), lr=0.0001)

    print(f'[training] train_data len = {len(train_data["train_captions"])}')
    print(f'[training] episodes = {episodes}')
    print(f'[training] epoch_count = {epoch_count}')

    for epoch in range(epoch_count):
        episodicAvgLoss = 0

        captions, features, _ = sample_coco_minibatch(train_data, batch_size=episodes, split='train')
        features = torch.tensor(features, device=device).float()
        captions = torch.tensor(captions, device=device).long()

        # decoded = decode_captions(captions, train_data['idx_to_word'])
        for episode in range(episodes):
            log_probs = []
            values = []
            rewards = []
            captions_in = captions[episode:episode + 1, :]
            features_in = features[episode:episode + 1]
            value, probs = a2cNetwork(features_in, captions_in)
            probs = F.softmax(probs, dim=2)

            dist = probs.cpu().detach().numpy()[0, 0]
            action = np.random.choice(probs.shape[-1], p=dist)

            gen_cap = torch.from_numpy(np.array([action])).unsqueeze(0).to(device)
            captions_in = torch.cat((captions_in, gen_cap), axis=1)

            log_prob = torch.log(probs[0, 0, action])

            reward = GetRewards(features_in, captions_in, rewardNet)
            reward = reward.cpu().detach().numpy()[0, 0]

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            values = torch.FloatTensor(values).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            log_probs = torch.stack(log_probs).to(device)

            advantage = values - rewards
            actorLoss = (-log_probs * advantage).mean()
            criticLoss = 0.5 * advantage.pow(2).mean()

            loss = actorLoss + criticLoss
            episodicAvgLoss += loss.item() / episodes

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[training] epoch:{epoch} episodicAvgLoss: {episodicAvgLoss}")

    torch.save(a2cNetwork.state_dict(), os.path.join(LOG_DIR, A2CNETWORK_WEIGHTS_FILE))
    results_filename = os.path.join(LOG_DIR, RESULTS_FILE)
    with open(results_filename, 'a') as f:
        f.write('\n' + '-' * 10 + ' network ' + '-' * 10 + '\n')
        f.write(str(a2cNetwork))
        f.write('\n' + '-' * 10 + ' network ' + '-' * 10 + '\n')

    return a2cNetwork


def test_a2cNetwork(a2cNetwork, test_data=None, data_size=None, validation_batch_size=100):
    a2cNetwork.train(False)
    real_captions_filename = os.path.join(LOG_DIR, REAL_CAPTIONS_FILE)
    generated_captions_filename = os.path.join(LOG_DIR, GENERATED_CAPTIONS_FILE)
    image_url_filename = os.path.join(LOG_DIR, IMAGE_URL_FILENAME)

    real_captions_file = open(real_captions_filename, "a")
    generated_captions_file = open(generated_captions_filename, "a")
    image_url_file = open(image_url_filename, "a")

    captions_real_all, features_real_all, urls_all = sample_coco_minibatch(test_data, batch_size=data_size, split='val')
    val_captions_lens = len(captions_real_all)
    loop_count = val_captions_lens // validation_batch_size
    for i in tqdm(range(loop_count), desc='Testing model'):
        captions_real = captions_real_all[i:i + validation_batch_size - 1]
        features_real = features_real_all[i:i + validation_batch_size - 1]
        urls = urls_all[i:i + validation_batch_size]

        captions_real_v = torch.tensor(captions_real, device=device).long()
        features_real_v = torch.tensor(features_real, device=device).float()

        value, probs = a2cNetwork(features_real_v, captions_real_v)
        probs = F.softmax(probs, dim=2)
        dist = probs.cpu().detach().numpy()[0, 0]
        action = np.random.choice(probs.shape[-1], p=dist)
        gen_cap = torch.from_numpy(np.array([action])).unsqueeze(0).to(device)
        gen_cap_str = decode_captions(gen_cap, idx_to_word=test_data["idx_to_word"])[0]
        real_cap_str = decode_captions(captions_real, idx_to_word=test_data["idx_to_word"])[0]

        real_captions_file.write(real_cap_str + '\n')
        generated_captions_file.write(gen_cap_str + '\n')
        image_url_file.write(urls[0] + '\n')

        # captions_real_v = captions_real_v.to('cpu')
        # features_real_v = features_real_v.to('cpu')
        del captions_real_v, features_real_v
        del gen_cap, value, probs, dist
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    real_captions_file.close()
    generated_captions_file.close()
    image_url_file.close()


def calculate_a2cNetwork_score():
    real_captions_filename = os.path.join(LOG_DIR, REAL_CAPTIONS_FILE)
    generated_captions_filename = os.path.join(LOG_DIR, GENERATED_CAPTIONS_FILE)

    ref, hypo = load_textfiles(real_captions_filename, generated_captions_filename)
    network_score = str(score(ref, hypo))
    print(network_score)

    results_filename = os.path.join(LOG_DIR, RESULTS_FILE)
    with open(results_filename, 'a') as f:
        f.write('\n' + '-' * 10 + ' results ' + '-' * 10 + '\n')
        f.write(network_score)
        f.write('\n' + '-' * 10 + ' results ' + '-' * 10 + '\n')


def init_deep_rl():
    global LOG_DIR, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print_green(f"[Info] Working on: {device}, device_name: {torch.cuda.get_device_name(0)} ")
    else:
        print_green(f"[Info] Working on: {device}")

    current_time_str = str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S"))
    LOG_DIR = os.path.join('logs', current_time_str)
    os.makedirs(LOG_DIR)


def main():
    init_deep_rl()

    max_train = None  # set None for whole traning dataset
    max_train_str = '' if max_train == None else str(max_train)
    print_green(f'[Info] Loading COCO dataset {max_train_str}')
    data = load_data(max_train=max_train, print_keys=True)
    print_green(f'[Info] COCO dataset loaded')

    print_green(f'[Info] Training A2C Network')
    a2cNetwork = train_a2cNetwork(train_data=data, epoch_count=10, episodes=50000)
    print_green(f'[Info] A2C Network trained')

    print_green(f'[Info] Testing A2C Network')
    test_a2cNetwork(a2cNetwork, test_data=data, data_size=500)
    print_green(f'[Info] A2C Network Tested')

    print_green(f'[Info] A2C Network score - start')
    calculate_a2cNetwork_score()
    print_green(f'[Info] A2C Network score - end')
    print_green(f'[Info] Logs saved in dir: {LOG_DIR}')


if __name__ == "__main__":
    main()
