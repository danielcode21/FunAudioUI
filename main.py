'''
Author: Daniel
'''
import gradio as gr
import os
from funaudiowarp import SenseVoiceNode
from funaudiowarp import CosyVoiceSFTNode
from funaudio_utils.download_models import ModelDownloader

base_path = os.path.dirname(os.path.abspath(__file__))
model_downloader = ModelDownloader(os.path.join(base_path, "models"))

sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
def get_text_by_sensevoice(audio_input, use_fast_model, punc_segment):
    '''
    调用SenseVoice的语音识别接口
    '''
    #print(f'audio_input={audio_input}, use_fast_model={use_fast_model}, punc_segment={punc_segment}')
    sensevoice_node = SenseVoiceNode(base_path, model_downloader)
    return sensevoice_node.generate(audio_input, use_fast_model, punc_segment)
def create_voice_by_predefined(tts_text, predefined_model, speed_input, seed, use_25hz_flag):
    '''
    调用CosyVoice的预定义模型生成语音
    '''
    #print(f'tts_text={tts_text}, predefined_model={predefined_model}, speed_input={speed_input}, seed={seed}, use_25hz_flag={use_25hz_flag}')
    cosyvoice_sft_node = CosyVoiceSFTNode(base_path, model_downloader)
    output = cosyvoice_sft_node.generate(tts_text, predefined_model, float(speed_input), int(seed), use_25hz_flag)
    return output
def create_voice_by_natural_lang(tts_text_for_natural_lang, instruct_text, predefined_model, speed_input, seed):
    '''
    调用CosyVoice的预定义模型和指令文本生成语音
    '''
    print(f'tts_text_for_natural_lang={tts_text_for_natural_lang}, instruct_text={instruct_text}, predefined_model={predefined_model}, speed_input={speed_input}, seed={seed}')

def create_voice_by_dual_lang(sample_audio, tts_text_for_dual_lang, speed_input, seed, use_25hz_flag):
    '''
    使用跨语言模型生成语音
    '''
    print(f'sample_audio={sample_audio}, tts_text_for_dual_lang={tts_text_for_dual_lang}, speed_input={speed_input}, seed={seed}, use_25hz_flag={use_25hz_flag}')
def create_voice_by_cloned_model(speaker_model_name, speaker_model_dir, tts_text, prompt_text, speed_input, seed, use_25hz_flag):
    '''
    使用克隆语音模型按钮
    '''
    print(f'speaker_model_name={speaker_model_name}, speaker_model_dir={speaker_model_dir}, tts_text={tts_text}, prompt_text={prompt_text}, speed_input={speed_input}, seed={seed}, use_25hz_flag={use_25hz_flag}')

def create_voice_model(sample_audio, tts_text, prompt_text, speed_input, seed, use_25hz_flag):
    '''
    生成语音克隆模型
    '''
    print(f'sample_audio={sample_audio}, tts_text={tts_text}, prompt_text={prompt_text}, speed_input={speed_input}, seed={seed}, use_25hz_flag={use_25hz_flag}')

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
                                 
def cosyvoice_predefined_model_node():
    '''
    CosyVoice预定义模型的节点布局
    '''
    with gr.Column('CosyVoice预定义模型'):
        with gr.Group():
            tts_text = gr.TextArea(label='TTS文本',lines=3)
        with gr.Group():
            predefined_model = gr.Dropdown(choices=sft_spk_list, label='预定义角色模型', value='中文女')
            speed_input = gr.Textbox(label='语速', value='1.0')
            seed = gr.Textbox(label='随机种子', value='1738')
            use_25hz_flag = gr.Checkbox(label='使用25hz', value=False)
   
    return tts_text, predefined_model, speed_input, seed, use_25hz_flag

def cosyvoice_natural_lang_control_node():
    '''
    CosyVoice自然语言控制节点布局
    '''
    with gr.Column('CosyVoice自然语言控制'):
        with gr.Group():
            tts_text = gr.TextArea(label='TTS文本',lines=3)
        with gr.Group():
            instruct_text = gr.TextArea(label='Instruct文本',lines=3)
        with gr.Group():
            predefined_model = gr.Dropdown(choices=sft_spk_list, label='预定义角色模型', value='中文女')
            speed_input = gr.Textbox(label='语速', value='1.0')
            seed = gr.Textbox(label='随机种子', value='1738')
    
    return tts_text, instruct_text, predefined_model, speed_input, seed

def cosyvoice_dual_lang_clone_node():
    '''
    CosyVoice跨语言克隆节点布局
    '''
    with gr.Column('CosyVoice跨语言克隆'):
        with gr.Group():
            sample_audio = gr.Audio(label='采样音频文件')
        
        with gr.Group():
            tts_text = gr.TextArea(label='TTS文本',lines=3)
        
        with gr.Group():
            speed_input = gr.Textbox(label='语速', value='1.0')
            seed = gr.Textbox(label='随机种子', value='1738')
            use_25hz_flag = gr.Checkbox(label='使用25hz', value=False)
    
    return sample_audio, tts_text, speed_input, seed, use_25hz_flag

def cosyvoice_fast_clone_node(mode):
    '''
    CosyVoice快速克隆节点布局
    mode: model 模型模式，model 模型模式
    '''
    if mode == 'model':
        mode_hide = True
        file_hide = False
    else:
        mode_hide = False
        file_hide = True

    with gr.Column('CosyVoice快速克隆'):
        with gr.Group():
            tts_text = gr.TextArea(label='TTS文本',lines=3)

        with gr.Group():
            prompt_text = gr.TextArea(label='Prompt文本',lines=3)

        with gr.Group():
                speaker_model_name = gr.Textbox(label='语音模型名称', visible=mode_hide)
                speaker_model_dir = gr.Textbox(label='语音模型目录', visible=mode_hide)
        
        with gr.Group():
                sample_audio = gr.Audio(label='采样音频文件', visible=file_hide)

        
        with gr.Group():
            speed_input = gr.Textbox(label='语速', value='1.0')
            seed = gr.Textbox(label='随机种子', value='1738')
            use_25hz_flag = gr.Checkbox(label='使用25hz', value=False)
    
        return sample_audio, speaker_model_name, speaker_model_dir, tts_text, prompt_text, speed_input, seed, use_25hz_flag

def app_launcher():
    '''
    主界面布局
    '''
    
    with gr.Blocks() as app:
        gr.Markdown('''
        # 👋 Hello World!
        ''')

        with gr.Tab('语音识别'):
            gr.Markdown('''
            使用SenseVoice识别音频，生成文字
            ''')
            with gr.Row():
                with gr.Column():
                    audio_input, use_fast_model, punc_segment = sensevoice_layout()
                    get_text_btn = gr.Button('识别')
                with gr.Column():
                    text_ouput = gr.TextArea(label='识别结果', lines=7, show_label=True, show_copy_button=True)

        with gr.Tab('语音生成'):
            gr.Markdown('''
            使用CosyVoice的预训练模型生成语音
            ''')
            with gr.Row():
                with gr.Column():
                    tts_text_for_predefined, predefined_model, speed_input, seed, use_25hz_flag = cosyvoice_predefined_model_node()
                    create_voice_by_predefined_btn = gr.Button('生成')
                with gr.Column():
                    voice_predefined = gr.Audio(label='生成语音')

        with gr.Tab('自然语言语音生成'):
            gr.Markdown('''
            使用CosyVoice的预训练模型生成语音
            ''')
            with gr.Row():
                with gr.Column():
                    tts_text_for_natural_lang, instruct_text, predefined_model2, speed_input2, seed2 = cosyvoice_natural_lang_control_node()
                    create_voice_by_natural_lang_btn = gr.Button('生成')
                with gr.Column():
                    voice_natural = gr.Audio(label='生成语音')
        
        with gr.Tab('跨语种语音生成'):
            gr.Markdown('''
            使用采样音频和CosyVideo预定义模型生成语音
            ''')
            with gr.Row():
                with gr.Column():
                    sample_audio3, tts_text_for_dual_lang, speed_input3, seed3, use_25hz_flag3 = cosyvoice_dual_lang_clone_node()
                    create_voice_by_dual_lang_btn = gr.Button('生成')
                with gr.Column():
                    voice_dual_lang = gr.Audio(label='生成语音')

        mode = 'model'
        with gr.Tab('语音模型语音生成'):
            gr.Markdown('''
            通过CosyVoice，使用自定义的语音模型生成语音
            ''')
            with gr.Row():
                with gr.Column():
                    sample_audio4, speaker_model_name, speaker_model_dir, tts_text4, prompt_text4, speed_input4, seed4, use_25hz_flag4 = cosyvoice_fast_clone_node(mode)
                    create_voice_by_cloned_model_btn = gr.Button("生成")

                with gr.Column():
                    cloned_voice = gr.Audio(label="生成语音")

        mode = 'file'
        with gr.Tab('语音克隆模型'):
            gr.Markdown('''
            使用CosyVoice生成语音克隆模型
            ''')
            with gr.Row():
                with gr.Column():
                    sample_audio5, speaker_model_name5, speaker_model_dir5, tts_text5, prompt_text5, speed_input5, seed5, use_25hz_flag5 = cosyvoice_fast_clone_node(mode)
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
            fn=create_voice_by_predefined,
            inputs=[tts_text_for_predefined, predefined_model, speed_input, seed, use_25hz_flag], 
            outputs=voice_predefined
        )

        create_voice_by_natural_lang_btn.click(
            fn=create_voice_by_natural_lang,
            inputs=[tts_text_for_natural_lang, instruct_text, predefined_model2, speed_input2, seed2],
            outputs=voice_natural
        )

        create_voice_by_dual_lang_btn.click(
            fn=create_voice_by_natural_lang,
            inputs=[sample_audio3, tts_text_for_dual_lang, speed_input3, seed3, use_25hz_flag3],
            outputs=voice_dual_lang
        )
        
        create_voice_by_cloned_model_btn.click(
            fn=create_voice_by_cloned_model,
            inputs=[speaker_model_name, speaker_model_dir, tts_text4, prompt_text4, speed_input4, seed4, use_25hz_flag4],
            outputs=cloned_voice
        )

        create_voice_model_btn.click(
            fn=create_voice_model,
            inputs=[sample_audio5, tts_text5, prompt_text5, speed_input5, seed5, use_25hz_flag5],
            outputs=[voice_output6, voice_model6]
        )

    return app
def main():
    app = app_launcher()
    app.launch()

if __name__ == '__main__':
    main()