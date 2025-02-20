import os
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import shutil
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_path)


from typing import Optional
import torch
import yaml
import re
import gc
from numba import cuda

from scipy.io.wavfile import read, write
from PIL import Image
from config import Settings
from omegaconf import OmegaConf

from text_function import text_encode, text_decode
from config import Settings, text_default_settings
from model import load_model, get_tokenizer, get_model
from utils import SingleExampleOutput, check_dir

class TextBackend: # 不要改参数名称 这个应该是internal不是output
    def __init__(self, default_save_internal_dir='./output/', default_save_uploads_dir='./uploads/'):
        super().__init__()
        # 默认内部存储文件夹，在程序结束后会删除
        self.default_save_internal_dir = default_save_internal_dir
        self.default_save_uploads_dir = default_save_uploads_dir
        os.makedirs(self.default_save_internal_dir, exist_ok=True)
        os.makedirs(self.default_save_uploads_dir, exist_ok=True)

        self.settings = None
        self.model = None
        self.tokenizer = None
        self.default_stego_save_path = os.path.join(self.default_save_internal_dir, 'stego_en.txt')  # 用于默认存放生成的stego文件
        self.default_message_save_path = os.path.join(self.default_save_internal_dir, 'message_de.txt')  # 用于默认存放提取的secret文件
        self.stego_mimetype = self.message_mimetype = 'text/plain'
        self.message_file_path = None
        self.stego_file_path = None  # 用于读取提取时候用到的stego文件
        self.prompt_file_path = None
        self.key_file_path = None
        # self.config_file_path = None

    def module_init(self, config_file_path, device='cuda:0'):
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.task = config.get('task')
            self.model_name = config.get('model_name')
            self.top_p = config.get('top_p')
            self.length = config.get('length')
            self.win_len = config.get('win_len')
            self.compression_alg = config.get('compression_alg')
            self.batched_encode = config.get('batched_encode')
            self.with_context_start = config.get('with_context_start')

            self.settings = Settings(task = self.task,
                                     model_name = self.model_name,
                                     top_p = self.top_p,
                                     length = self.length,
                                     win_len = self.win_len,
                                     compression_alg = self.compression_alg,
                                     batched_encode = self.batched_encode,
                                     with_context_start = self.with_context_start,)
            # self.model = get_model(self.settings)
            # self.tokenizer = get_tokenizer(self.settings)
            self.model, self.tokenizer, self.compress_encoder, self.decompress_decoder = load_model(self.settings)
            return 200, '模型和算法已加载完成，请开始使用。'
        except Exception:
            return 400, '显卡和环境与模块不适配，出现错误！'

    def embedding(self):
        if self.model is None:
            return 400, '模型未成功加载，无法执行操作！'
        if self.tokenizer is None:
            return 400, '分词器未成功加载，无法执行操作！'
        if self.key_file_path is None:
            return 400, '您并未上传密钥文件，无法选择消息文件!'
        if self.message_file_path is None:
            return 400, '消息文件未上传，无法执行生成任务！'
        try:
            text_encode(self.settings, self.model, self.tokenizer, self.message_file_path, self.prompt_file_path, self.key_file_path)
        except Exception:
            return 400, '秘密信息嵌入过程中存在问题，请联系管理员排查。可尝试更换参数等。'
        return 200, '秘密信息嵌入过程已完成。'

    def extracting(self):
        if self.model is None:
            return 400, '模型未成功加载，无法执行操作！'
        if self.tokenizer is None:
            return 400, '分词器未成功加载，无法执行操作！'
        if self.key_file_path is None:
            return 400, '您并未上传密钥文件，无法选择消息文件!'
        if self.stego_file_path is None:
            return 400, '含密文件未上传，无法执行提取任务！'
        try:
            text_decode(self.settings, self.model, self.tokenizer, self.stego_file_path, self.prompt_file_path, self.key_file_path)
        except Exception:
            return 400, '秘密信息提取过程中存在问题，您输入的文本无法解密。'
        return 200, '秘密信息提取过程已完成。'

    def embedding_download(self):
        return self.default_stego_save_path, self.stego_mimetype

    def extracting_download(self):
        return self.default_message_save_path, self.message_mimetype

    def prompt_checker(self, prompt_file):
        prompt_file_path = os.path.join(self.default_save_uploads_dir, prompt_file.filename)
        prompt_file.save(prompt_file_path)
        message_code, message_str = self._prompt_checker(prompt_file_path)
        return message_code, message_str

    def prompt_embedding_checker(self, prompt_file):
        return self.prompt_checker(prompt_file)

    def prompt_extracting_checker(self, prompt_file):
        return self.prompt_checker(prompt_file)

    def key_checker(self, key_file):
        key_file_path = os.path.join(self.default_save_uploads_dir, key_file.filename)
        key_file.save(key_file_path)
        message_code, message_str = self._key_checker(key_file_path)
        return message_code, message_str

    def key_embedding_checker(self, key_file):
        return self.key_checker(key_file)

    def key_extracting_checker(self, key_file):
        return self.key_checker(key_file)

    def message_file_checker(self, message_file):
        message_file_path = os.path.join(self.default_save_uploads_dir, message_file.filename)
        message_file.save(message_file_path)
        message_code, message_str = self._message_file_checker(message_file_path)
        return message_code, message_str

    def stego_file_checker(self, stego_file):
        stego_file_path = os.path.join(self.default_save_uploads_dir, stego_file.filename)
        stego_file.save(stego_file_path)
        message_code, message_str = self._stego_file_checker(stego_file_path)
        return message_code, message_str

    def _contains_chinese(self, prompt_str):
        return re.search(r'[\u4e00-\u9fff]', prompt_str) is not None

    def _prompt_checker(self, prompt_file_path):
        if self.model is None:
            return 400, '模型未成功加载，无法执行操作！'
        if self.tokenizer is None:
            return 400, '分词器未成功加载，无法执行操作！'
        if not self._is_txt(prompt_file_path):
            return 400, '您上传的提示词文件格式有误，目前只支持文本文件。'
        try:
            with open(prompt_file_path, 'r', encoding="utf-8") as f:
                prompt_str = f.read().strip()
        except Exception:
            return 400, '文件受损或不存在，请重新检查'
        if self._contains_chinese(prompt_str):
            return 400, '算法目前并不支持中文。'
        message_str = '已成功记录提示词。'

        self.prompt_filnie_path = prompt_file_path
        return 200, message_str

    # def _is_png(self, file_path):
    #     ext = os.path.splitext(file_path)[1].lower()
    #     return ext == '.png'

    def _is_txt(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext == '.txt'

    # def _is_jpeg(self, file_path):
    #     ext = os.path.splitext(file_path)[1].lower()
    #     return ext in ['.jpg', '.jpeg']

    def _key_checker(self, key_file_path):
        if self.model is None:
            return 400, '模型未成功加载，无法执行操作！'
        if self.tokenizer is None:
            return 400, '分词器未成功加载，无法执行操作！'

        self.key_file_path = key_file_path
        return 200, '密钥文件已成功加载。'

    def _message_file_checker(self, message_file_path):
        if self.model is None:
            return 400, '模型未成功加载，无法执行操作！'
        if self.tokenizer is None:
            return 400, '分词器未成功加载，无法执行操作！'
        if self.key_file_path is None:
            return 400, '您并未上传密钥文件，无法选择消息文件!'
        if not self._is_txt(message_file_path):
            return 400, '您上传的消息文件格式有误，目前只支持文本（TXT）文件。'
        self.message_file_path = message_file_path
        return 200, '消息文件已成功加载。'

    def _stego_file_checker(self, stego_file_path):
        if self.model is None:
            return 400, '模型未成功加载，无法执行操作！'
        if self.tokenizer is None:
            return 400, '分词器未成功加载，无法执行操作！'
        if self.key_file_path is None:
            return 400, '您并未上传密钥文件，无法选择消息文件!'
        if not self._is_txt(stego_file_path):
            return 400, '您上传的含密文件格式有误，目前只支持TXT文本文件。'
        self.stego_file_path = stego_file_path
        return 200, '含密文件已成功加载。'

    def clear_directory(self, directory_path):
        # 检查路径是否存在
        if os.path.exists(directory_path):
            # 遍历目录中的所有文件和子目录
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                # 删除文件
                if os.path.isfile(item_path):
                    os.remove(item_path)
                # 删除子目录及其内容
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    # 关闭当前后端占用的显存
    def close(self):
        self.clear_directory(self.default_save_internal_dir)
        self.clear_directory(self.default_save_uploads_dir)
        device_num = self.model.device.index
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        cuda.select_device(device_num)
        cuda.close()


if __name__ == '__main__':
    subb = TextBackend()
    res = subb.module_init('./settings.yaml')
    print(res[-1])

    res = subb._prompt_checker('../uploads/prompt.txt')
    print(res[-1])
    res = subb._key_checker('../uploads/seed.txt')
    print(res[-1])
    # res = subb._message_file_checker('C:\\Users\\Admin\\Desktop\\00000_selected\\00012.png')
    # print(res[-1])
    # res = subb.embedding()
    # print(res[-1])
    res = subb._stego_file_checker('../uploads/stego_en.txt')
    print(res[-1])
    res = subb.extracting()
    print(res[-1])
    # res = subb.extracting()
    # print(res[-1])
    # 以下代码可以清除显存

    del subb
    torch.cuda.empty_cache()
    print('12346')
    # import time
    # time.sleep(120)