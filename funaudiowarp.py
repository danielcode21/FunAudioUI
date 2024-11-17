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
import librosa
from funasr import AutoModel
from funasr.utils import postprocess_utils
from funaudio_utils.cosyvoice_plus import CosyVoicePlus
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav
from time import time as ttime

class SenseVoiceNode:
    '''
    SenseVoice功能
    '''
    def __init__(self, base_path, model_downloader):
        self.__base_path = base_path
        self.__model_downloader = model_downloader
        self.__prompt_sr = 16000
    def __patch_emoji(self, emoji_dict):
        t_emoji_dict_key = emoji_dict.keys()
        emoji_dict_new = {}
        for t_e_k in t_emoji_dict_key:
            emoji_dict_new[t_e_k.lower()] = emoji_dict[t_e_k]
        emoji_dict.update(emoji_dict_new)
        return emoji_dict

    def __audio_resampling(self, audio):
        fs, speech = audio
        speech = speech.astype(np.float32) / np.iinfo(np.int16).max
        if len(speech.shape) > 1:
            speech = speech.mean(-1)
        if fs != self.__prompt_sr :
            print(f"audio_fs: {fs}")
            resampler = torchaudio.transforms.Resample(fs, self.__prompt_sr )
            speech_t = torch.from_numpy(speech).to(torch.float32)
            speech = resampler(speech_t[None, :])[0, :].numpy()

        return speech

    def generate(self, audio, use_fast_mode, punc_segment):
        sensevoice_code_path = os.path.join(self.__base_path,"funaudiollm/sensevoice/model.py")
        #print(f'sensevoice_code_path={sensevoice_code_path}')
        #print(audio)
    
        if isinstance(audio, tuple):
            speech = self.__audio_resampling(audio)
        # if isinstance(audio, tuple):
        #     fs, speech = audio
        #     speech = speech.astype(np.float32) / np.iinfo(np.int16).max
        #     if len(speech.shape) > 1:
        #         speech = speech.mean(-1)
        #     if fs != self.__prompt_sr :
        #         print(f"audio_fs: {fs}")
        #         resampler = torchaudio.transforms.Resample(fs, self.__prompt_sr )
        #         speech_t = torch.from_numpy(speech).to(torch.float32)
        #         speech = resampler(speech_t[None, :])[0, :].numpy()

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

class CosyVoiceNode:
    '''
    CosyVoice功能
    '''
    def __init__(self, base_path, model_downloader, target_sr=22500, prompt_sr=16000):
        self._base_path = base_path
        self._model_downloader = model_downloader
        self._target_sr = target_sr
        self._prompt_sr = prompt_sr
        self._max_val = 0.8
    def _return_audio(self, output, t0, spk_model):
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
        audio = (self._target_sr, waveform)
        if spk_model is not None:
            return (audio,spk_model,)
        else:
            return audio
        
    def _postprocess(self, speech, top_db=60, hop_length=220, win_length=440):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > self._max_val:
            speech = speech / speech.abs().max() * self._max_val
        speech = torch.concat([speech, torch.zeros(1, int(self._target_sr * 0.2))], dim=1)
        return speech
    
    def generate(self):
        pass

    def _load_speaker_from_path(self, speaker_name, model_dir):
        spk_model_path = os.path.join(model_dir, speaker_name + ".pt")
        assert os.path.exists(spk_model_path), "Speaker model is not exist"
        spk_model = torch.load(os.path.join(model_dir, speaker_name + ".pt"),map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return spk_model

class CosyVoiceSFTNode(CosyVoiceNode):
    '''
    CosyVoice功能
    '''
    def __init__(self, base_path, model_downloader):
        super().__init__(base_path, model_downloader)

    def generate(self, tts_text, speaker_name, speed, seed, use_25hz):
        t0 = ttime()
        target_sr = 22050
        _, model_dir = self._model_downloader.download_cosyvoice_300m_sft(use_25hz)
        cosyvoice = CosyVoicePlus(model_dir)
        set_all_random_seed(seed)
        generator = cosyvoice.inference_sft(tts_text, speaker_name, False, speed)
        return self._return_audio(generator, t0, None)
        
class CosyVoiceNaturalNode(CosyVoiceNode):
    '''
    CosyVoice功能
    '''
    def __init__(self, base_path, model_downloader):
        super().__init__(base_path, model_downloader)
    def generate(self, tts_text, instruct_text, speaker_name, speed, seed):
        t0 = ttime()
        _, model_dir = self._model_downloader.download_cosyvoice_300m_instruct()
        cosyvoice = CosyVoicePlus(model_dir)
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct(tts_text, speaker_name, instruct_text, False, speed)
        return self._return_audio(output,t0,None)

class CosyVoiceDualLanglNode(CosyVoiceNode):
    '''
    CosyVoice功能
    '''
    def __init__(self, base_path, model_downloader):
        super().__init__(base_path, model_downloader)

    def generate(self, audio, tts_text, speed, seed, use_25hz):        
        t0 = ttime()
        _, model_dir = self._model_downloader.download_cosyvoice_300m(use_25hz)
        cosyvoice = CosyVoicePlus(model_dir)
        speech = load_wav(audio, self._prompt_sr)
        prompt_speech_16k = self._postprocess(speech)
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, False, speed)
        return self._return_audio(output, t0, None)

class CosyVoiceSpeakerVoiceNode(CosyVoiceNode):
    '''
    CosyVoice功能
    '''
    def __init__(self, base_path, model_downloader):
        super().__init__(base_path, model_downloader)
    def generate(self, speaker_name, speaker_dir, tts_text, speed, seed, use_25hz):
        t0 = ttime()
        _, model_dir = self._model_downloader.download_cosyvoice_300m(use_25hz)
        cosyvoice = CosyVoicePlus(model_dir)
        speaker_model = self._load_speaker_from_path(speaker_name, speaker_dir)
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot_with_spkmodel(tts_text, speaker_model, False, speed)
        return self._return_audio(output, t0, None)
    
class CosyVoiceSpeakerCreaterNode(CosyVoiceNode):
    '''
    CosyVoice功能
    '''
    def __init__(self, base_path, model_downloader):
        super().__init__(base_path, model_downloader)
    def generate(self, audio, tts_text, prompt_text, speaker_name, speaker_dir, speed, seed, use_25hz):
        t0 = ttime()
        _, model_dir = self._model_downloader.download_cosyvoice_300m(use_25hz)
        cosyvoice = CosyVoicePlus(model_dir)
        assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
        speech = load_wav(audio, self._prompt_sr)
        prompt_speech_16k = self._postprocess(speech)
        #print('get zero_shot inference request')
        #print(model_dir)
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, False, speed)
        spk_model = cosyvoice.frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k)
        del spk_model['text']
        del spk_model['text_len']
        sample_audio, speaker = self._return_audio(output, t0, spk_model)

        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)
        # 保存模型
        speaker_path = os.path.join(speaker_dir, speaker_name + ".pt")
        torch.save(speaker, speaker_path)
        return (sample_audio, speaker_path)