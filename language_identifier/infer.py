import logging
from typing import List, Union

import torch
import numpy as np

from speechbrain.pretrained import EncoderClassifier

class LanguageIdentifier:
    def __init__(self, config):

        self.device: Union[List[int], int] = [0] if (torch.cuda.is_available() and config.num_gpus > 0) else 1
        self.accelerator: str = 'gpu' if (torch.cuda.is_available() and config.num_gpus > 0) else 'cpu'
        self.map_location: str = torch.device(f'cuda:{self.device[0]}') if self.accelerator == 'gpu' else 'cpu'

        self.model = EncoderClassifier.from_hparams(source=config.model_dir)
        self.model = self.model.eval()

    ''' Main prediction function '''
    def predict(self, audio_tensor: Union[np.ndarray, torch.tensor]) -> str:
        
        if type(audio_tensor) is np.ndarray:
            audio_tensor = torch.tensor(audio_tensor)
        
        elif type(audio_tensor) is not torch.tensor:
            raise TypeError('Input is not an np array or tensor')
        
        preds = self.model.classify_batch(audio_tensor)
        lang = preds[3]

        return lang
