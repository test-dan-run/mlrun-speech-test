import base64
from typing import Dict, Any, Union, Tuple

import numpy as np

from infer import Standardizer
from configs import std_config as config

class MRStandardizer:

    def __init__(self, context, name: Union[None, str] = None, **kwargs):

        self.context = context
        self.name = name
        self.kw = kwargs

        self.std = Standardizer(config)
    
    def read_request(self, json: Dict[str, Any]) -> Tuple[np.ndarray, int]:

        audio_bytes = json['audio']
        sample_rate = json['rate']
        channels = json['channels']

        msg = base64.b64decode(audio_bytes)

        audio = np.frombuffer(msg, dtype=np.int16)
        audio = audio.astype(np.float32, order='C') / 32767

        # reshape if multi-channeled
        audio = audio.reshape((len(audio)//channels, channels))

        return audio, sample_rate
    
    def do(self, x):

        audio, sample_rate = self.read_request(x)
        output_audio = self.std.convert(audio, sample_rate)

        return output_audio
