import logging
# make deterministic
import os

import datetime

from mingpt.utils import set_seed
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
import torch
import argparse

import experience_dataset
from experience_dataset import ExperienceDataset
from experience_dataset import Experience

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--game', type=str, default='LunarLander-v2')
parser.add_argument('--batch_size', type=int, default=128)
#
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
args = parser.parse_args()

set_seed(args.seed)


def save_model(net):
    time = datetime.datetime.now()
    time = time.strftime("%b%m_%H-%M-%S")

    model_path = "./checkpoints"

    cwd = os.getcwd()
    model_path = os.path.join(cwd, model_path)
    try:
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
    except OSError as error:
        print(error)

    model_path = os.path.join(model_path, 'ckpt_{}.pth'.format(time))

    torch.save(net.state_dict(), model_path)


def load_replays(path):
    cwd = os.path.join(os.getcwd())
    replay_path = os.path.join(cwd, './replays')

    try:
        if not os.path.isdir(replay_path):
            os.makedirs(replay_path)
    except OSError as error:
        print(error)

    replay_path = os.path.join(replay_path, path)

    return experience_dataset.load_data(replay_path)


# Load the saved replays.
cwd = os.path.join(os.getcwd(), os.pardir)
cwd = os.path.join(cwd, os.pardir)
cwd = os.path.join(cwd, "lunar_lander/data/LunarLander.npz")

data = experience_dataset.load_data(cwd)
states, actions, rtgs, done_idxs, timesteps = experience_dataset.create_dataset(data)

# set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

train_dataset = ExperienceDataset(states, actions, rtgs, done_idxs, timesteps, args.context_length * 3)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=3, n_head=4, n_embd=128, model_type=args.model_type, max_timestep=int(max(timesteps)))

model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=256 * 20,
                      final_tokens=2 * len(train_dataset) * args.context_length * 3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game,
                      max_timestep=max(timesteps))
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()

save_model(model)
