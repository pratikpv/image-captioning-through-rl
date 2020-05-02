import os, json, h5py, sys, time, requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
from datetime import datetime
from metrics import *
from tqdm import tqdm

# defaults and params
device = "cuda"
data = {}
MAX_SEQ_LEN = 17
BASE_DIR = 'datasets/coco_captioning'
REAL_CAPTIONS_FILE = 'real_captions.txt'
GENERATED_CAPTIONS_FILE = 'generated_captions.txt'
IMAGE_URL_FILENAME = 'image_url.txt'
LOG_DIR = ""
A2CNETWORK_WEIGHTS_FILE = 'a2cNetwork.pt'
RESULTS_FILE = 'results.txt'


def print_green(text):
    print('\033[32m', text, '\033[0m', sep='')


def print_red(text):
    print('\033[31m', text, '\033[0m', sep='')


def load_data(base_dir=BASE_DIR, max_train=None, pca_features=True, print_keys=False):
    caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
    else:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
    with h5py.File(train_feat_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    if pca_features:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
    else:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
    with h5py.File(val_feat_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    data["train_captions_lens"] = np.zeros(data["train_captions"].shape[0])
    data["val_captions_lens"] = np.zeros(data["val_captions"].shape[0])
    for i in range(data["train_captions"].shape[0]):
        data["train_captions_lens"][i] = np.nonzero(data["train_captions"][i] == 2)[0][0] + 1
    for i in range(data["val_captions"].shape[0]):
        data["val_captions_lens"][i] = np.nonzero(data["val_captions"][i] == 2)[0][0] + 1

    if print_keys:
        # Print out all the keys and values from the data dictionary
        for k, v in data.items():
            if type(v) == np.ndarray:
                print(k, type(v), v.shape, v.dtype)
            else:
                print(k, type(v), len(v))

    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
    split_size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idxs' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls


def get_coco_validation_data(data, data_size=None):
    captions = data['val_captions']
    image_features = data['val_features']
    urls = data['val_urls']
    return captions, image_features, urls


def image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


class PolicyNetwork(nn.Module):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=512, hidden_dim=512, dtype=np.float32):
        super(PolicyNetwork, self).__init__()

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)

        self.cnn2linear = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(wordvec_dim, hidden_dim, batch_first=True)
        self.linear2vocab = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        input_captions = self.caption_embedding(captions)
        hidden_init = self.cnn2linear(features)
        cell_init = torch.zeros_like(hidden_init)
        output, _ = self.lstm(input_captions, (hidden_init, cell_init))
        output = self.linear2vocab(output)
        return output


class ValueNetworkRNN(nn.Module):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=512, hidden_dim=512, dtype=np.float32):
        super(ValueNetworkRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        vocab_size = len(word_to_idx)

        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_dim).to(device), torch.zeros(1, 1, self.hidden_dim).to(device))

        self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)
        self.lstm = nn.LSTM(wordvec_dim, hidden_dim)

    def forward(self, captions):
        input_captions = self.caption_embedding(captions)
        output, self.hidden_cell = self.lstm(input_captions.view(len(input_captions), 1, -1), self.hidden_cell)
        return output


class ValueNetwork(nn.Module):
    def __init__(self, word_to_idx):
        super(ValueNetwork, self).__init__()
        self.valrnn = ValueNetworkRNN(word_to_idx)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, features, captions):
        for t in range(captions.shape[1]):
            vrnn = self.valrnn(captions[:, t])
        vrnn = vrnn.squeeze(0).squeeze(1)
        state = torch.cat((features, vrnn), dim=1)
        output = self.linear1(state)
        output = self.linear2(output)
        return output


class RewardNetworkRNN(nn.Module):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=512, hidden_dim=512, dtype=np.float32):
        super(RewardNetworkRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        vocab_size = len(word_to_idx)

        self.hidden_cell = torch.zeros(1, 1, self.hidden_dim).to(device)

        self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)
        self.gru = nn.GRU(wordvec_dim, hidden_dim)

    def forward(self, captions):
        input_captions = self.caption_embedding(captions)
        output, self.hidden_cell = self.gru(input_captions.view(len(input_captions), 1, -1), self.hidden_cell)
        return output


class RewardNetwork(nn.Module):
    def __init__(self, word_to_idx):
        super(RewardNetwork, self).__init__()
        self.rewrnn = RewardNetworkRNN(word_to_idx)
        self.visual_embed = nn.Linear(512, 512)
        self.semantic_embed = nn.Linear(512, 512)

    def forward(self, features, captions):
        # TODO: Pratik. Why assign to rrnn in loop?
        for t in range(captions.shape[1]):
            rrnn = self.rewrnn(captions[:, t])
        rrnn = rrnn.squeeze(0).squeeze(1)
        se = self.semantic_embed(rrnn)
        ve = self.visual_embed(features)
        return ve, se


def GenerateCaptions(features, captions, model):
    features = torch.tensor(features, device=device).float().unsqueeze(0)
    gen_caps = torch.tensor(captions[:, 0:1], device=device).long()
    for t in range(MAX_SEQ_LEN - 1):
        output = model(features, gen_caps)
        gen_caps = torch.cat((gen_caps, output[:, -1:, :].argmax(axis=2)), axis=1)
    return gen_caps


class AdvantageActorCriticNetwork(nn.Module):
    def __init__(self, valueNet, policyNet):
        super(AdvantageActorCriticNetwork, self).__init__()

        self.valueNet = valueNet
        self.policyNet = policyNet

    def forward(self, features, captions):
        # Get value from value network
        values = self.valueNet(features, captions)
        # Get action probabilities from policy network
        probs = self.policyNet(features.unsqueeze(0), captions)[:, -1:, :]
        return values, probs


def GetRewards(features, captions, model):
    visEmbeds, semEmbeds = model(features, captions)
    visEmbeds = F.normalize(visEmbeds, p=2, dim=1)
    semEmbeds = F.normalize(semEmbeds, p=2, dim=1)
    rewards = torch.sum(visEmbeds * semEmbeds, axis=1).unsqueeze(1)
    return rewards


def train_a2cNetwork(train_data=None, epoch_count=10, episodes=100):
    rewardNet = RewardNetwork(data["word_to_idx"]).to(device)
    policyNet = PolicyNetwork(data["word_to_idx"]).to(device)
    valueNet = ValueNetwork(data["word_to_idx"]).to(device)

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

        # decoded = decode_captions(captions, data['idx_to_word'])
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


def test_a2cNetwork(a2cNetwork, data=None, data_size=None, validation_batch_size=100):
    a2cNetwork.train(False)
    real_captions_filename = os.path.join(LOG_DIR, REAL_CAPTIONS_FILE)
    generated_captions_filename = os.path.join(LOG_DIR, GENERATED_CAPTIONS_FILE)
    image_url_filename = os.path.join(LOG_DIR, IMAGE_URL_FILENAME)

    real_captions_file = open(real_captions_filename, "a")
    generated_captions_file = open(generated_captions_filename, "a")
    image_url_file = open(image_url_filename, "a")

    captions_real_all, features_real_all, urls_all = sample_coco_minibatch(data, batch_size=data_size, split='val')
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
        gen_cap_str = decode_captions(gen_cap, idx_to_word=data["idx_to_word"])[0]
        real_cap_str = decode_captions(captions_real, idx_to_word=data["idx_to_word"])[0]

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
    a2cNetwork = train_a2cNetwork(train_data=data, epoch_count=10, episodes=1)
    print_green(f'[Info] A2C Network trained')

    print_green(f'[Info] Testing A2C Network')
    test_a2cNetwork(a2cNetwork, data=data, data_size=500)
    print_green(f'[Info] A2C Network Tested')

    print_green(f'[Info] A2C Network score - start')
    calculate_a2cNetwork_score()
    print_green(f'[Info] A2C Network score - end')
    print_green(f'[Info] Logs saved in dir: {LOG_DIR}')


if __name__ == "__main__":
    main()
    # calculate_a2cNetwork_score()
