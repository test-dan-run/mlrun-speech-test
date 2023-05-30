import logging

import sox
import numpy as np

class Standardizer:
    def __init__(self, config):

        self.tfm = sox.Transformer()
        self.tfm.set_output_format(rate=config.target_rate, channels=config.target_channels)

    ''' Main conversion function '''
    def convert(self, audio_array: np.ndarray, input_rate: int) -> np.ndarray:
        
        if type(audio_array) is not np.ndarray:
            raise TypeError('Input is not an np array')
        
        output_audio_array = self.tfm.build_array(input_array=audio_array, sample_rate_in=input_rate)

        return output_audio_array
