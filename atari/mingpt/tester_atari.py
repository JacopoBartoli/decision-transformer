import logging

import os

import datetime

import torch

from gym.wrappers.monitoring import video_recorder

from .trainer_atari import Env, Args
from .utils import sample

logger = logging.getLogger(__name__)


def create_video_path(env_name):
    """
    Create the path hierarchy.
    :param env_name: the name of the environment.
    :return: the recording path
    """
    time = datetime.datetime.now()
    time = time.strftime("%b%m_%H-%M-%S")

    video_path = "./video/{}_{}".format(env_name, time)
    cwd = os.getcwd()
    video_path = os.path.join(cwd, video_path)
    try:
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
    except OSError as error:
        print(error)

    recorder_path = os.path.join(video_path, "{}".format(env_name))

    return recorder_path


class TesterConfig:
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Tester:

    def __init__(self, model, config, recordings=False):
        self.model = model
        self.config = config
        self.recordings = recordings

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        if recordings:
            self.recordings_path = create_video_path(config.game)

    def test(self, ret=0):
        self.model.train(False)
        args = Args(self.config.game, self.config.seed)
        env = Env(args)
        env.eval()

        if self.recordings:
            vid = video_recorder.VideoRecorder(env.env, base_path=self.recordings_path)

        state = env.reset()
        done = False
        reward_sum = 0
        state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
        rtgs = [ret]
        # first state is from env, first rtg is target return, and first timestep is 0
        sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None,
                                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(
                                    -1),
                                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

        j = 0
        all_states = state
        actions = []
        while not done:
            if self.recordings:
                vid.capture_frame()

            action = sampled_action.cpu().numpy()[0, -1]
            actions += [sampled_action]
            state, reward, done = env.step(action)
            reward_sum += reward
            j += 1

            state = state.unsqueeze(0).unsqueeze(0).to(self.device)

            all_states = torch.cat([all_states, state], dim=0)

            rtgs += [rtgs[-1] - reward]
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep
            sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(
                                        1).unsqueeze(0),
                                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(
                                        0).unsqueeze(-1),
                                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1),
                                                                                             dtype=torch.int64).to(
                                        self.device)))
        env.close()
        if self.recordings:
            vid.close()
        test_return = reward_sum
        # print("target return: %d, eval return: %d" % (ret, test_return))
        # self.model.train(True)

        return test_return