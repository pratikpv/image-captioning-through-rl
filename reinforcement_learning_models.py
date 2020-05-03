import time
import torch.optim as optim

from tqdm import tqdm
from utility_functions import *
from reinforcement_learning_networks import *


def train_value_network():
    return

def train_policy_network():
    return

def train_reward_network():
    return

def train_a2c_network(train_data, save_paths, epoch_count=10, episodes=100):
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
        
        episode_t = time.time()
        for episode in range(episodes):

            captions_in = captions[episode:episode + 1, :]
            features_in = features[episode:episode + 1]

            values, probs = a2cNetwork(features_in, captions_in)

            probs = F.softmax(probs, dim=2)
            dist = probs.cpu().detach().numpy()[0, 0]
            action = np.random.choice(probs.shape[-1], p=dist)

            gen_cap = torch.from_numpy(np.array([action])).unsqueeze(0).to(device)
            try:
                captions_in = torch.cat((captions_in, gen_cap), axis=1)
            except:
                captions_in = torch.cat((captions_in, gen_cap.long()), axis=1)

            log_probs = torch.log(probs[0, 0, action])

            rewards = GetRewards(features_in, captions_in, rewardNet)
            rewards = rewards.cpu().detach().numpy()[0, 0]

            values = torch.FloatTensor([values]).to(device)
            rewards = torch.FloatTensor([rewards]).to(device)
            log_probs = torch.stack([log_probs]).to(device)

            advantage = values - rewards
            actorLoss = (-log_probs * advantage).mean()
            criticLoss = 0.5 * advantage.pow(2).mean()

            loss = actorLoss + criticLoss
            episodicAvgLoss += loss.item() / episodes

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if episode % 1000 == 0:
                print("[training] episode: %s, time taken: %ss" % (episode, time.time() - episode_t))
                print("[training] current memory allocated: %s\t | cached memory: %s" \
                                % (torch.cuda.memory_allocated() / 1024 ** 2, \
                                        torch.cuda.memory_cached() / 1024 ** 2))
                # print_garbage_collection()
                episode_t = time.time()
        
        print(f"[training] epoch:{epoch} episodicAvgLoss: {episodicAvgLoss}")
        torch.cuda.empty_cache()

    model_save_path = save_paths["model_path"]
    results_save_path = save_paths["results_path"]

    torch.save(a2cNetwork.state_dict(), model_save_path)
    with open(results_save_path, 'a') as f:
        f.write('\n' + '-' * 10 + ' network ' + '-' * 10 + '\n')
        f.write(str(a2cNetwork))
        f.write('\n' + '-' * 10 + ' network ' + '-' * 10 + '\n')

    return a2cNetwork


def test_a2c_network(a2cNetwork, test_data, image_caption_data, data_size=None, validation_batch_size=100):

    a2cNetwork.train(False)

    real_captions_filename = image_caption_data["real_captions_path"]
    generated_captions_filename = image_caption_data["generated_captions_path"]
    image_url_filename = image_caption_data["image_urls_path"]

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
