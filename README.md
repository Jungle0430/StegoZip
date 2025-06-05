# StegoZip
StegoZip

## Overview


## Usage.
1. Create folder.
```bash
$ mkdir checkpoint data result
$ cd data && mkdir compress origin && cd ..
```

2. Create environment.
```bash
$ conda create -n stegozip python=3.9.21 -y
$ conda activate stegozip
$ pip install -r requirements.txt
$ pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
$ python -m spacy download en_core_web_md
$ cd src && python stega/setup.py build_ext --build-lib=stega/ && cd ..
```

3. Download Dataset.
```bash
$ mv download_ag_news.py data/
$ cd data && python download_ag_news.py && cd ..
```

4. Run the demo script.
```bash
$ chmod +x run.sh
$ nohup ./run.sh >> main_result.txt
```

5. Wait for the task to complete. If any error occurs during the process, it will be reported in the `main_result.txt` file. Upon successful completion, the final results will be displayed in the `result/` folder.

## Acknowledgment

## Citation
If you find this work or repository useful, please consider citing or starring it.