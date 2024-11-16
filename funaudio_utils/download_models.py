'''
Orginal Author: SpenserCai
Date: 2024-10-04 13:54:01
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-04 22:26:22
Description: file content
'''
import modelscope
import os
from modelscope import snapshot_download

# Download the model
class ModelDownloader:
    def __init__(self, base_model_path):
        self.__base_cosyvoice_model_path = os.path.join(base_model_path, "CosyVoice")
        self.__base_sensevoice_model_path = os.path.join(base_model_path, "SenseVoice")

    def download_cosyvoice_300m(self, is_25hz=False):
        model_name = "CosyVoice-300M"
        model_id = "iic/CosyVoice-300M"
        if is_25hz:
            model_name = "CosyVoice-300M-25Hz"
            model_id = "iic/CosyVoice-300M-25Hz"
        model_dir = os.path.join(self.__base_cosyvoice_model_path, model_name)
        snapshot_download(model_id=model_id, local_dir=model_dir)
        return model_name, model_dir

    def download_cosyvoice_300m_sft(self, is_25hz=False):
        model_name = "CosyVoice-300M-SFT"
        model_id = "iic/CosyVoice-300M-SFT"
        if is_25hz:
            model_name = "CosyVoice-300M-SFT-25Hz"
            model_id = "MachineS/CosyVoice-300M-SFT-25Hz"
        model_dir = os.path.join(self.__base_cosyvoice_model_path, model_name)
        snapshot_download(model_id=model_id, local_dir=model_dir)
        return model_name, model_dir

    def download_sensevoice_small(self):
        model_name = "SenseVoiceSmall"
        model_id = "iic/SenseVoiceSmall"
        model_dir = os.path.join(self.__base_sensevoice_model_path, model_name)
        snapshot_download(model_id=model_id, local_dir=model_dir)
        return model_name, model_dir

    def download_cosyvoice_300m_instruct(self):
        model_name = "CosyVoice-300M-Instruct"
        model_id = "iic/CosyVoice-300M-Instruct"
        model_dir = os.path.join(self.__base_cosyvoice_model_path, model_name)
        snapshot_download(model_id=model_id, local_dir=model_dir)
        return model_name, model_dir

    def get_speaker_default_path(self):
        return os.path.join(self.__base_cosyvoice_model_path, "Speaker")