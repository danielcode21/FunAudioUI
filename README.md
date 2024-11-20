这个项目是参考了[ComfyUI-FunAudioLLM](https://github.com/SpenserCai/ComfyUI-FunAudioLLM)和[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)这两个项目改写的。主要就是做了一个外壳，将调用封装起来了。

因为主要目地是自用，所以其中深入一些的内容还没有研究，也没有充分测试（开发环境：mac + vscode + conda），界面也只有中文。

requirements.txt目前还没有确认，基本是照搬**ComfyUI0FunAudioLLM**这个项目的。

使用方法：
1. git clone这个项目
2. 进入clone的目录，安装python虚拟环境
3. 安装requirements.txt
4. 运行：python app.py

补充:
1. 初次使用某个功能时会从网上下载模型（我用的是参考项目的代码，看起来是从[modelscope 魔搭](https://www.modelscope.cn/)下载的），都下载下来就很占流量和空间了。
2. 自然语言生成的指令文本可以参考[官网](https://fun-audio-llm.github.io/#CosyVoice-instructed)和[网友终结](https://note.youdao.com/ynoteshare/index.html?id=aacd7b7c3229a35a4fd6a6a82976eba7&type=note&_time=1732058153538)
3. 可以使用 python app.py -n 0.0.0.0 -p 8080 指定启动参数