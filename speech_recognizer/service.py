import base64
from typing import Dict, Any, Union

import numpy as np

from infer import SpeechRecognizer
from configs import asr_config as config

class MRSpeechRecognizer:

    def __init__(self, context, name: Union[None, str] = None, **kwargs):

        self.context = context
        self.name = name
        self.kw = kwargs

        self.model = SpeechRecognizer(config)
    
    def read_request(self, json: Dict[str, Any]) -> np.ndarray:

        audio_bytes = json['audio']
        msg = base64.b64decode(audio_bytes)

        audio = np.frombuffer(msg, dtype=np.int16)
        audio = audio.astype(np.float32, order='C') / 32767

        return audio
    
    def do(self, x):

        audio = self.read_request(x)
        text = self.model.predict(audio)

        return text
