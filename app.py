'''
Author: Daniel
'''
import gradio as gr
import os
import random
from funaudiowarp import SenseVoiceNode
from funaudiowarp import CosyVoiceSFTNode, CosyVoiceNaturalNode, CosyVoiceDualLanglNode, CosyVoiceSpeakerVoiceNode, CosyVoiceSpeakerCreaterNode
from funaudio_utils.download_models import ModelDownloader

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "models")
speaker_path = os.path.join(model_path, 'Speaker')
model_downloader = ModelDownloader(model_path)

sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
def get_text_by_sensevoice(audio_input, use_fast_model, punc_segment):
    '''
    调用SenseVoice的语音识别接口
    '''
    sensevoice_node = SenseVoiceNode(base_path, model_downloader)
    return sensevoice_node.generate(audio_input, use_fast_model, punc_segment)

def create_voice_by_pretrained(tts_text, predefined_model, speed_input, seed, use_25hz_flag):
    '''
    调用CosyVoice的预定义模型生成语音
    '''
    cosyvoice_sft_node = CosyVoiceSFTNode(base_path, model_downloader)
    output = cosyvoice_sft_node.generate(tts_text, predefined_model, float(speed_input), int(seed), use_25hz_flag)
    return output

def create_voice_by_natural_lang(tts_text_for_natural_lang, instruct_text, predefined_model, speed_input, seed):
    '''
    调用CosyVoice的预定义模型和指令文本生成语音
    '''
    cosyvoice_natural_node = CosyVoiceNaturalNode(base_path, model_downloader)
    output = cosyvoice_natural_node.generate(tts_text_for_natural_lang, instruct_text, predefined_model, float(speed_input), int(seed))
    return output

def create_voice_by_dual_lang(sample_audio, tts_text_for_dual_lang, speed_input, seed, use_25hz_flag):
    '''
    使用跨语言模型生成语音
    '''
    cosyvoice_dual_lang_node = CosyVoiceDualLanglNode(base_path, model_downloader)
    output = cosyvoice_dual_lang_node.generate(sample_audio, tts_text_for_dual_lang, float(speed_input), int(seed), use_25hz_flag)
    return output

def create_voice_by_cloned_model(speaker_model_name, speaker_model_dir, tts_text, speed_input, seed, use_25hz_flag):
    '''
    使用克隆语音模型按钮
    '''
    cosyvoice_speaker_voice_node = CosyVoiceSpeakerVoiceNode(base_path, model_downloader)
    output = cosyvoice_speaker_voice_node.generate(speaker_model_name, speaker_model_dir, tts_text, float(speed_input), int(seed), use_25hz_flag)
    return output

def create_voice_model(sample_audio, speaker_model_name, speaker_model_dir, tts_text, prompt_text, speed_input, seed, use_25hz_flag):
    '''
    生成语音克隆模型
    '''
    cosyvoice_speaker_create_node = CosyVoiceSpeakerCreaterNode(base_path, model_downloader)
    output = cosyvoice_speaker_create_node.generate(sample_audio, tts_text, prompt_text, speaker_model_name, speaker_model_dir, float(speed_input), int(seed), use_25hz_flag)
    return output

def generate_seed():
    '''
    生成随机种子
    '''
    seed = random.randint(1, 100000000)
    return seed

def common_param_layout():
    '''
    通用的参数布局
    '''
    with gr.Column('CosyVoice克隆模型'):
        with gr.Row():
            with gr.Column():
                speed_input = gr.Slider(label="语速", minimum=0.5, maximum=2.0, step=0.1, value=1.0, min_width=100)
            with gr.Group():
                with gr.Column():
                    seed = gr.Number(label='随机种子', value=generate_seed(),  min_width=100)
                with gr.Column():
                    seed_btn = gr.Button("🎲", min_width=20)
            
        with gr.Row():
            use_25hz_flag = gr.Checkbox(label='使用25Hz采样率', value=False)

    seed_btn.click(
        fn=generate_seed,
        inputs=[],
        outputs=[seed]
    )

    return speed_input, seed, seed_btn, use_25hz_flag
 
def sensevoice_layout():
    '''
    SenseVoice节点布局
    '''
    with gr.Column('SenseVoice'):
        with gr.Group():
            audio_input = gr.Audio(label='源音频文件')
        with gr.Group():
            use_fast_model = gr.Checkbox(label='使用快速模型', value=False)
            punc_segment = gr.Checkbox(label='添加标点符号', value=True)
    
    return audio_input, use_fast_model, punc_segment
                                 
def cosyvoice_pretrained_model_node():
    '''
    CosyVoice预定义模型的节点布局
    '''
    with gr.Column('CosyVoice预定义模型'):
        with gr.Group():
            tts_text = gr.TextArea(label='TTS文本',lines=5)
        with gr.Group():
            predefined_model = gr.Dropdown(choices=sft_spk_list, label='预定义角色模型', value='中文女')
            speed_input, seed, seed_btn, use_25hz_flag = common_param_layout()
   
    return tts_text, predefined_model, speed_input, seed, seed_btn, use_25hz_flag

def cosyvoice_natural_lang_control_node():
    '''
    CosyVoice自然语言控制节点布局
    '''
    with gr.Column('CosyVoice自然语言控制'):
        with gr.Group():
            tts_text = gr.TextArea(label='TTS文本',lines=5)
        with gr.Group():
            instruct_text = gr.TextArea(label='Instruct文本',lines=3)
        with gr.Group():
            predefined_model = gr.Dropdown(choices=sft_spk_list, label='预定义角色模型', value='中文女')
            speed_input, seed, seed_btn, _ = common_param_layout()
    
    return tts_text, instruct_text, predefined_model, speed_input, seed, seed_btn

def cosyvoice_dual_lang_clone_node():
    '''
    CosyVoice跨语言克隆节点布局
    '''
    with gr.Column('CosyVoice跨语言克隆'):
        with gr.Group():
            sample_audio = gr.Audio(label='采样音频文件', type='filepath')
        
        with gr.Group():
            tts_text = gr.TextArea(label='TTS文本',lines=5)
        
        with gr.Group():
            speed_input, seed, seed_btn, use_25hz_flag = common_param_layout()
    
    return sample_audio, tts_text, speed_input, seed, seed_btn, use_25hz_flag

def cosyvoice_fast_clone_node(mode):
    '''
    CosyVoice快速克隆节点布局
    mode: model 模型模式，model 模型模式
    '''
    if mode == 'model':
        file_hide = False
    else:
        file_hide = True

    with gr.Column('CosyVoice快速克隆'):
        with gr.Group():
            tts_text = gr.TextArea(label='TTS文本',lines=3)

        with gr.Group():
            prompt_text = gr.TextArea(label='Prompt文本',lines=3, visible=file_hide)

        with gr.Group():
            speaker_model_name = gr.Textbox(label='语音模型名称')
            speaker_model_dir = gr.Textbox(label='语音模型目录', value=speaker_path)
        
        with gr.Group():
            sample_audio = gr.Audio(label='采样音频文件', type='filepath', visible=file_hide)

        with gr.Group():
            speed_input, seed, seed_btn, use_25hz_flag = common_param_layout()
    
        return sample_audio, speaker_model_name, speaker_model_dir, tts_text, prompt_text, speed_input, seed, seed_btn, use_25hz_flag

def app_launcher():
    '''
    主界面布局
    '''
    
    with gr.Blocks() as app:
        gr.Markdown('''
        # 👋 FunAudioLLM TTS工具
        ''')

        with gr.Tab('语音识别'):
            gr.Markdown('''
            使用SenseVoice识别音频，提取文字。
            step1. 上传音频文件；
            step2. 设置参数；
            step3. 点击识别按钮，提取文字。
            ''')
            with gr.Row():
                with gr.Column():
                    audio_input, use_fast_model, punc_segment = sensevoice_layout()
                    get_text_btn = gr.Button('识别')
                with gr.Column():
                    text_ouput = gr.TextArea(label='识别结果', lines=9, show_label=True, show_copy_button=True)

        with gr.Tab('语音生成(预训练模型)'):
            gr.Markdown('''
            使用CosyVoice的预训练模型生成语音。
            step1. 输入TTS文本；
            step2. 选择预训练的角色模型；
            step3. 设置语速等参数；
            step4. 点击生成按钮，生成语音。
            ''')
            with gr.Row():
                with gr.Column():
                    tts_text_for_predefined, predefined_model, speed_input, seed, seed_btn, use_25hz_flag = cosyvoice_pretrained_model_node()
                    create_voice_by_predefined_btn = gr.Button('生成')
                with gr.Column():
                    voice_predefined = gr.Audio(label='生成语音')

        with gr.Tab('自然语言控制语音生成'):
            gr.Markdown('''
            使用CosyVoice的预训练模型生成语音。
            step1. 输入TTS文本；
            step2. 输入Instruct文本，用于描述角色的说话特点；
            step3. 选择预训练的角色模型；
            step4. 设置语速等参数；
            step5. 点击生成按钮，生成语音。
            ''')
            with gr.Row():
                with gr.Column():
                    tts_text_for_natural_lang, instruct_text, predefined_model2, speed_input2, seed2, seed_btn2 = cosyvoice_natural_lang_control_node()
                    create_voice_by_natural_lang_btn = gr.Button('生成')
                with gr.Column():
                    voice_natural = gr.Audio(label='生成语音')
        
        with gr.Tab('跨语种语音生成'):
            gr.Markdown('''
            使用采样音频和CosyVideo预定义模型生成语音。
            step1. 上传音频文件，确定音色；
            step2. 输入TTS文本；
            step3. 设置语速等参数；
            step4. 点击生成按钮，生成语音。
            ''')
            with gr.Row():
                with gr.Column():
                    sample_audio3, tts_text_for_dual_lang, speed_input3, seed3, seed_btn3, use_25hz_flag3 = cosyvoice_dual_lang_clone_node()
                    create_voice_by_dual_lang_btn = gr.Button('生成')
                with gr.Column():
                    voice_dual_lang = gr.Audio(label='生成语音')

        mode = 'model'
        with gr.Tab('语音模型语音生成'):
            gr.Markdown('''
            通过CosyVoice，使用自训练的语音模型生成语音。
            step1. 输入TTS文本；
            step2. 输入模型的名称和目录；
            step3. 设置语速等参数；
            step4. 点击生成按钮，生成语音。
            ''')
            with gr.Row():
                with gr.Column():
                    speaker_audio4, speaker_model_name, speaker_model_dir, tts_text4, prompt_text4, speed_input4, seed4, seed_btn4, use_25hz_flag4 = cosyvoice_fast_clone_node(mode)
                    create_voice_by_cloned_model_btn = gr.Button("生成")

                with gr.Column():
                    cloned_voice = gr.Audio(label="生成语音")

        mode = 'file'
        with gr.Tab('语音克隆模型'):
            gr.Markdown('''
            使用CosyVoice生成语音克隆模型。
            step1. 输入TTS文本，用于生成模型的测试语音；
            step2. 输入Prompt文本，这个需要和step3上传的音频文件内容一致；
            step3. 上传音频文件，是用于生成模型的源音频；
            step4. 设置语速等参数；
            step5. 点击生成按钮，生成模型和测试语音。
            ''')
            with gr.Row():
                with gr.Column():
                    sample_audio5, speaker_model_name5, speaker_model_dir5, tts_text5, prompt_text5, speed_input5, seed5, seed_btn5, use_25hz_flag5 = cosyvoice_fast_clone_node(mode)
                    create_voice_model_btn = gr.Button("生成")

                with gr.Column():
                    voice_output6 = gr.Audio(label="生成语音")
                    voice_model6 = gr.File(label="生成语音模型")

        get_text_btn.click(
            fn=get_text_by_sensevoice,
            inputs=[audio_input, use_fast_model, punc_segment],
            outputs=text_ouput
        )

        create_voice_by_predefined_btn.click(
            fn=create_voice_by_pretrained,
            inputs=[tts_text_for_predefined, predefined_model, speed_input, seed, use_25hz_flag], 
            outputs=voice_predefined
        )

        create_voice_by_natural_lang_btn.click(
            fn=create_voice_by_natural_lang,
            inputs=[tts_text_for_natural_lang, instruct_text, predefined_model2, speed_input2, seed2],
            outputs=voice_natural
        )

        create_voice_by_dual_lang_btn.click(
            fn=create_voice_by_dual_lang,
            inputs=[sample_audio3, tts_text_for_dual_lang, speed_input3, seed3, use_25hz_flag3],
            outputs=voice_dual_lang
        )
        
        create_voice_by_cloned_model_btn.click(
            fn=create_voice_by_cloned_model,
            inputs=[speaker_model_name, speaker_model_dir, tts_text4, speed_input4, seed4, use_25hz_flag4],
            outputs=cloned_voice
        )

        create_voice_model_btn.click(
            fn=create_voice_model,
            inputs=[sample_audio5, speaker_model_name5, speaker_model_dir5, tts_text5, prompt_text5, speed_input5, seed5, use_25hz_flag5],
            outputs=[voice_output6, voice_model6]
        )

    return app

def main():
    app = app_launcher()
    app.launch()

if __name__ == '__main__':
    main()