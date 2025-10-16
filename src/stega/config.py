import os
import torch


class Settings:
    def __init__(self,
                 task: str = 'text',
                 algo: str = 'Discop',
                 model_name: str = 'LLaMA-7B',
                 temp: float = 0.9,
                 top_p: float = 0.92,
                 length: int = 100,
                 win_len: int = 511,
                 compression_alg: str = '',
                 batched_encode: bool = False,
                 with_context_start: bool = False,
                 seed: int = os.urandom(5),
                 repetition_penalty: float = 1.2,
                 output_path = '../text_output',
                 device=torch.device('cuda')):

        if task not in ['text', 'image', 'text-to-speech']:
            raise NotImplementedError("`Settings.task` must belong to {'text', 'image', 'text-to-speech'}!")
        self.task = task
        self.algo = algo
        self.model_name = model_name

        if temp is None:
            temp = 1.0
        self.temp = temp

        if top_p is None:
            top_p = 1.0
        elif top_p <= 0 or top_p > 1:
            raise ValueError('`top_p` must be in (0, 1]!')
        self.top_p = top_p

        self.length = length
        self.seed = seed
        self.output_path = output_path
        self.device = device
        self.win_len = win_len
        self.compression_alg = compression_alg
        self.batched_encode = batched_encode
        self.with_context_start = with_context_start
        self.repetition_penalty = repetition_penalty

    def __call__(self):
        return self.algo, self.temp, self.top_p, self.length, self.seed

    def __str__(self):
        return '\n'.join('{} = {}'.format(key, value) for (key, value) in self.__dict__.items())

text_default_settings_discop = Settings('text',
                                        model_name='LLaMA-7B',
                                        algo='Discop',
                                        top_p=1.0,
                                        length=1000,
                                        win_len=511,
                                        batched_encode=False,
                                        with_context_start=False)

text_default_settings_sparsamp = Settings('text',
                                        model_name='LLaMA-7B',
                                        algo='SparSamp',
                                        top_p=1.0,
                                        length=1000,
                                        win_len=511,
                                        batched_encode=False,
                                        with_context_start=False)

text_default_settings_meteor = Settings('text',
                                        model_name='LLaMA-7B',
                                        algo='Meteor',
                                        top_p=1.0,
                                        length=1000,
                                        win_len=511,
                                        batched_encode=False,
                                        with_context_start=False)

text_default_settings_sample = Settings('text',
                                        model_name='LLaMA-7B',
                                        algo='sample',
                                        top_p=1.0,
                                        length=200,
                                        win_len=511,
                                        batched_encode=False,
                                        with_context_start=False)
