'''
Author: Daniel
Date: 2023-08-05 11:55:46
Description: Use this module to wrap the FunAudioLLM.
Refer: FunaudioLLM node files
'''
import os
import numpy as np
import torch
import torchaudio
from funasr import AutoModel
#from funaudiollm.funaudio_utils.pre import FunAudioLLMTool
from funaudio_utils.download_models import ModelDownloader
from funasr.utils import postprocess_utils
from funaudio_utils.cosyvoice_plus import CosyVoicePlus
from cosyvoice.utils.common import set_all_random_seed
from time import time as ttime

class SenseVoiceNode:
    '''
    SenseVoice功能
    '''
    def __init__(self, base_path, model_downloader):
        self.__base_path = base_path
        #print('base_model_path=', base_model_path)
        self.__model_downloader = model_downloader
        #self.__fAudioTool = FunAudioLLMTool()
    def __patch_emoji(self, emoji_dict):
        t_emoji_dict_key = emoji_dict.keys()
        emoji_dict_new = {}
        for t_e_k in t_emoji_dict_key:
            emoji_dict_new[t_e_k.lower()] = emoji_dict[t_e_k]
        emoji_dict.update(emoji_dict_new)
        return emoji_dict

    def generate(self, audio, use_fast_mode, punc_segment):
        sensevoice_code_path = os.path.join(self.__base_path,"funaudiollm/sensevoice/model.py")
        #print(f'sensevoice_code_path={sensevoice_code_path}')
        #print(audio)
    
        if isinstance(audio, tuple):
            fs, speech = audio
            speech = speech.astype(np.float32) / np.iinfo(np.int16).max
            if len(speech.shape) > 1:
                speech = speech.mean(-1)
            if fs != 16000:
                print(f"audio_fs: {fs}")
                resampler = torchaudio.transforms.Resample(fs, 16000)
                speech_t = torch.from_numpy(speech).to(torch.float32)
                speech = resampler(speech_t[None, :])[0, :].numpy()

        # 判断语音长度是否大于30s
        if speech.shape[0] > 30 * 22050 and use_fast_mode:
            raise ValueError("Audio length is too long, please set use_fast_mode to False.")
        _, model_dir = self.__model_downloader.download_sensevoice_small()
        model_arg = {
                "input":speech,
                "cache":{},
                "language":"auto",
                "batch_size_s":60,
        }
        model_use_arg = {
            "model":model_dir,
            "trust_remote_code":True,
            "remote_code":sensevoice_code_path,
            "device":"cuda:0",
        }

        if not use_fast_mode:
            model_use_arg["vad_model"] = "fsmn-vad"
            model_use_arg["vad_kwargs"] = {"max_single_segment_time":30000}

            model_arg["merge_vad"] = True
            model_arg["merge_length_s"] = 15

        if punc_segment:
            model_use_arg["punc_model"] = "ct-punc-c"
        
        model = AutoModel(**model_use_arg)
        output = model.generate(**model_arg)
        postprocess_utils.emoji_dict = self.__patch_emoji(postprocess_utils.emoji_dict)
        
        return postprocess_utils.rich_transcription_postprocess(output[0]["text"])

class CosyVoiceSFTNode:
    '''
    CosyVoice功能
    '''
    def __init__(self, base_path, model_downloader):
        self.__base_path = base_path
        self.__model_downloader = model_downloader

    def __return_audio(self, output, t0, spk_model, target_sr):
        output_list = []
        for out_dict in output:
            output_numpy = out_dict['tts_speech'].squeeze(0).numpy() * 32768 
            output_numpy = output_numpy.astype(np.int16)
            #output_list.append(torch.Tensor(output_numpy/32768).unsqueeze(0))
            output_list.append(output_numpy)
        t1 = ttime()
        print("cost time \t %.3f" % (t1-t0))
        #audio = {"waveform": torch.cat(output_list,dim=1).unsqueeze(0),"sample_rate":target_sr}
        waveform = np.concatenate(output_list, axis=0).astype(np.float32) / 32768  # 正规化到 [-1, 1]
        audio = (target_sr, waveform)
        if spk_model is not None:
            return (audio,spk_model,)
        else:
            return audio

    def generate(self, tts_text, speaker_name, speed, seed, use_25hz):
        t0 = ttime()
        target_sr = 22050
        _, model_dir = self.__model_downloader.download_cosyvoice_300m_sft(use_25hz)
        cosyvoice = CosyVoicePlus(model_dir)
        set_all_random_seed(seed)
        generator = cosyvoice.inference_sft(tts_text, speaker_name, False, speed)
        return self.__return_audio(generator, t0, None, target_sr)
        
    