import os
import json
import torch
import bert_score
import subprocess
import numpy as np
from argparse import Namespace
from rouge_score import rouge_scorer
from sacremoses import MosesTokenizer
from src.p_sp_utils.models import load_model
from src.p_sp_utils.evaluate_sts import Example
from sentence_transformers import SentenceTransformer, util

class FileSim(object):
    def __init__(self):
        self.similarity = lambda s1, s2: np.nan_to_num(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))

    def score(self, params, batcher, input1, input2, use_sent_transformers=False):
        sys_scores = []
        if not use_sent_transformers:
            for ii in range(0, len(input1), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            for i in range(len(input1)):
                embedding_1 = model.encode(input1[i], convert_to_tensor=True)
                embedding_2 = model.encode(input2[i], convert_to_tensor=True)
                score = util.pytorch_cos_sim(embedding_1, embedding_2)
                sys_scores.append(score.item())
        return sys_scores

def batcher(params, batch):
    new_batch = []
    for p in batch:
        if params.tokenize:
            tok = params.entok.tokenize(p, escape=False)
            p = " ".join(tok)
        if params.lower_case:
            p = p.lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        p = Example(p, params.lower_case)
        p.populate_embeddings(params.model.vocab, params.model.zero_unk, params.model.ngrams)
        new_batch.append(p)
    x, l = params.model.torchify_batch(new_batch)
    vecs = params.model.encode(x, l)
    return vecs.detach().cpu().numpy()

class PSPCalculator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.model, self.entok, self.new_args = self._init_model()
    
    def _init_model(self):
        args = {
            'gpu': 1 if torch.cuda.is_available() else 0,
            'load_file': 'src/p_sp_utils/paraphrase-at-scale-english/model.para.lc.100.pt',
            'sp_model': 'src/p_sp_utils/paraphrase-at-scale-english/paranmt.model',
        }
        
        if not os.path.exists(args['load_file']) or not os.path.exists(args['sp_model']):
            ALT_PATH = "src/p_sp_utils"
            if os.path.exists(os.path.join(ALT_PATH, os.path.basename(args['load_file']))):
                args['load_file'] = os.path.join(ALT_PATH, os.path.basename(args['load_file']))
                args['sp_model'] = os.path.join(ALT_PATH, os.path.basename(args['sp_model']))
            else:
                print("Downloading pretrained models...")
                os.makedirs('src/p_sp_utils', exist_ok=True)
                subprocess.run(['wget', 'http://www.cs.cmu.edu/~jwieting/paraphrase-at-scale-english.zip'])
                subprocess.run(['unzip', 'paraphrase-at-scale-english.zip', '-d', 'src/p_sp_utils'])
                os.remove('paraphrase-at-scale-english.zip')

        model, _ = load_model(None, args)
        model.eval()
        entok = MosesTokenizer(lang='en')
        
        new_args = Namespace(
            batch_size=32,
            entok=entok,
            sp=model.sp,
            params=args,
            model=model,
            lower_case=model.args.lower_case,
            tokenize=model.args.tokenize
        )
        return model, entok, new_args

    def calculate_psp(self, texts1, texts2):
        sim = FileSim()
        return sim.score(self.new_args, batcher, texts1, texts2)

def clean_text(text):
    """Clean whitespace and newlines in text"""
    return " ".join(text.strip().split())

def compute_rouge(scorer, reference, generated):
    """Calculate ROUGE metrics including precision, recall and f-measure"""
    reference = clean_text(reference)
    generated = clean_text(generated)
    scores = scorer.score(reference, generated)
    return {
        "rouge1": {
            "precision": scores["rouge1"].precision,
            "recall": scores["rouge1"].recall,
            "fmeasure": scores["rouge1"].fmeasure,
        },
        "rouge2": {
            "precision": scores["rouge2"].precision,
            "recall": scores["rouge2"].recall,
            "fmeasure": scores["rouge2"].fmeasure,
        },
        "rougeL": {
            "precision": scores["rougeL"].precision,
            "recall": scores["rougeL"].recall,
            "fmeasure": scores["rougeL"].fmeasure,
        },
    }

def evaluation(input_file, output_file):
    with open(input_file, "r") as f:
        test_data = json.load(f)
    
    # calculate bert score
    original_texts = [clean_text(s['original_text'].replace("[", "").replace("]", "")) for s in test_data]
    restored_texts = [clean_text(s['restored_text'].replace("[", "").replace("]", "")) for s in test_data]
    compressed_texts = [clean_text(s['compressed_text'].replace("[", "").replace("]", "")) for s in test_data]

    P_restored, R_restored, F1_restored = bert_score.score(
        cands=restored_texts,
        refs=original_texts,
        lang='en',
        # model_type='bert-base-uncased',
        model_type='roberta-large',
        verbose=True
    )

    P_compressed, R_compressed, F1_compressed = bert_score.score(
        cands=compressed_texts,
        refs=original_texts,
        lang='en',
        # model_type='bert-base-uncased',
        model_type='roberta-large',
        verbose=True
    )

    for i, sample in enumerate(test_data):
        sample['bert_scores'] = {
            'original_restored': {
                'precision': float(P_restored[i].numpy()),
                'recall': float(R_restored[i].numpy()),
                'f1': float(F1_restored[i].numpy())
            },
            'original_compressed': {
                'precision': float(P_compressed[i].numpy()),
                'recall': float(R_compressed[i].numpy()),
                'f1': float(F1_compressed[i].numpy())
            }
        }

    avg_bert = {
        'original_restored': {
            'precision': float(np.mean(P_restored.numpy())),
            'recall': float(np.mean(R_restored.numpy())),
            'f1': float(np.mean(F1_restored.numpy()))
        },
        'original_compressed': {
            'precision': float(np.mean(P_compressed.numpy())),
            'recall': float(np.mean(R_compressed.numpy())),
            'f1': float(np.mean(F1_compressed.numpy()))
        }
    }
    
    average_metrics = {
        'bert_score': avg_bert
    }
    
    # cauculate psp
    psp_calculator = PSPCalculator()
    original_compressed_pairs = [(ori, com) for ori, com in zip(original_texts, compressed_texts)]
    original_restored_pairs = [(ori, res) for ori, res in zip(original_texts, restored_texts)]
    
    psp_oc = psp_calculator.calculate_psp(
        [p[0] for p in original_compressed_pairs],
        [p[1] for p in original_compressed_pairs]
    )
    psp_or = psp_calculator.calculate_psp(
        [p[0] for p in original_restored_pairs],
        [p[1] for p in original_restored_pairs]
    )

    for i, sample in enumerate(test_data):
        sample['psp_scores'] = {
            'original_compressed': np.float64(psp_oc[i]),
            'original_restored': np.float64(psp_or[i])
        }
    
    avg_oc = np.mean(psp_oc)
    avg_or = np.mean(psp_or)
    
    average_metrics['psp'] = {
            'original_compressed': np.float64(avg_oc),
            'original_restored': np.float64(avg_or)
    }
    
    # calculate rouge
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_results = []
    
    for i, sample in enumerate(test_data):
        rouge_restored = compute_rouge(scorer, sample['original_text'], sample['restored_text'])
        rouge_compressed = compute_rouge(scorer, sample['original_text'], sample['compressed_text'])
        sample['rouge'] = {
            'rouge_restored': rouge_restored,
            'rouge_compressed': rouge_compressed
        }
        
        rouge_results.append({
            'rouge_restored': rouge_restored,
            'rouge_compressed': rouge_compressed
        })
    
    avg_restored = {
        "rouge1": {
            "precision": sum(r["rouge_restored"]["rouge1"]["precision"] for r in rouge_results) / len(test_data),
            "recall": sum(r["rouge_restored"]["rouge1"]["recall"] for r in rouge_results) / len(test_data),
            "fmeasure": sum(r["rouge_restored"]["rouge1"]["fmeasure"] for r in rouge_results) / len(test_data),
        },
        "rouge2": {
            "precision": sum(r["rouge_restored"]["rouge2"]["precision"] for r in rouge_results) / len(test_data),
            "recall": sum(r["rouge_restored"]["rouge2"]["recall"] for r in rouge_results) / len(test_data),
            "fmeasure": sum(r["rouge_restored"]["rouge2"]["fmeasure"] for r in rouge_results) / len(test_data),
        },
        "rougeL": {
            "precision": sum(r["rouge_restored"]["rougeL"]["precision"] for r in rouge_results) / len(test_data),
            "recall": sum(r["rouge_restored"]["rougeL"]["recall"] for r in rouge_results) / len(test_data),
            "fmeasure": sum(r["rouge_restored"]["rougeL"]["fmeasure"] for r in rouge_results) / len(test_data),
        },
    }
    
    avg_compressed = {
        "rouge1": {
            "precision": sum(r["rouge_compressed"]["rouge1"]["precision"] for r in rouge_results) / len(test_data),
            "recall": sum(r["rouge_compressed"]["rouge1"]["recall"] for r in rouge_results) / len(test_data),
            "fmeasure": sum(r["rouge_compressed"]["rouge1"]["fmeasure"] for r in rouge_results) / len(test_data),
        },
        "rouge2": {
            "precision": sum(r["rouge_compressed"]["rouge2"]["precision"] for r in rouge_results) / len(test_data),
            "recall": sum(r["rouge_compressed"]["rouge2"]["recall"] for r in rouge_results) / len(test_data),
            "fmeasure": sum(r["rouge_compressed"]["rouge2"]["fmeasure"] for r in rouge_results) / len(test_data),
        },
        "rougeL": {
            "precision": sum(r["rouge_compressed"]["rougeL"]["precision"] for r in rouge_results) / len(test_data),
            "recall": sum(r["rouge_compressed"]["rougeL"]["recall"] for r in rouge_results) / len(test_data),
            "fmeasure": sum(r["rouge_compressed"]["rougeL"]["fmeasure"] for r in rouge_results) / len(test_data),
        },
    }
    
    average_metrics['rouge'] = {
        'restored': avg_restored,
        'compressed': avg_compressed
    }
    
    # calculate payload
    payload = []
    for sample in test_data:
        sample_payload = round(len(sample['original_text']) / len(sample['stego']['stego_text']), 6)
        sample['payload'] = sample_payload
        payload.append(sample_payload)
    
    average_metrics['payload'] = round(np.mean(payload), 6)
    
    # calculate time
    time1 = np.mean([sample["compress_time_cost"] for sample in test_data])
    time2 = np.mean([sample["encode_time_cost"] for sample in test_data])
    time3 = np.mean([sample["stego"]['encode_time'] for sample in test_data])
    time4 = np.mean([sample["stego"]['decode_time'] for sample in test_data])
    time5 = np.mean([sample["decode_time_cost"] for sample in test_data])
    time6 = np.mean([sample["restore_time_cost"] for sample in test_data])
    time7 = np.mean([sample["check_time"] for sample in test_data])
    
    encode_time = time1 + time2 + time3 + time7
    decode_time = time4 + time5 + time6
    average_metrics['time'] = {
        'DSRP': round(time1, 6),
        'Check': round(time7, 6),
        'ICC': round(time2, 6),
        'Embed': round(time3, 6),
        'Extract': round(time4, 6),
        'De-ICC': round(time5, 6),
        'Restore': round(time6, 6),
        'encode_time': round(encode_time, 6),
        'decode_time': round(decode_time, 6),
        'total_time': round(encode_time + decode_time, 6)
    }
    
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(average_metrics, f, ensure_ascii=False, indent=4)