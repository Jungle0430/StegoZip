# StegoZip
StegoZip

## How to run (a demo).
1. Create folder.
```bash
$ mkdir checkpoint data result
$ cd data && mkdir compress origin
```

2. Download Dataset.
```bash
$ mv ../download_ag_news.py ./
$ python download_ag_news.py && cd ..
```

3. Create environment.
```bash
$ conda create -n stegozip python=3.9.21
$ conda activate stegozip
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_md
$ cd src && python stego/setup.py build_ext --build-lib=stego/ && cd ..
```

4. Run the demo script.
```bash
$ chmod +x run.sh
$ nohup ./run.sh >> main_result.txt
```

5. Wait for the task to complete. If an error occurs, it will be reported in main_result.txt. Upon successful completion, the final results will be shown in the "result" folder.