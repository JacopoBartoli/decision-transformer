import logging

import os

import argparse

import torch

from tqdm import tqdm

from mingpt.utils import set_seed
from mingpt.model_atari import GPT, GPTConfig

from mingpt.tester_atari import Tester, TesterConfig

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--game', type=str, default='LunarLander-v2')
parser.add_argument('--vocab_size', type=int, default=4)
parser.add_argument('--max_timestep', type=int, default=1000)
parser.add_argument('--num_test', type=int, default=100)
#
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--checkpoint_file', type=str, default='ckpt')
args = parser.parse_args()

set_seed(args.seed)

mconf = GPTConfig(args.vocab_size, args.context_length * 3,
                  n_layer=3, n_head=4, n_embd=128, model_type='naive', max_timestep=args.max_timestep)
model = GPT(mconf)

# Load the model weights.
cwd = os.getcwd()
path = os.path.join(cwd, args.checkpoint_dir)

path = os.path.join(path, args.checkpoint_file)

model.load_state_dict(torch.load(path))

tconf = TesterConfig(seed=args.seed, game=args.game, max_timestep=args.max_timestep)

# If the testing video recordings is needed set the values to True.
tester = Tester(model, tconf, recordings=False)

result = 0
for i in tqdm(range(args.num_test)):
    result += tester.test()
result = result / args.num_test

print("Mean reward: {}".format(result))
