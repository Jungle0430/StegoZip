from datasets import load_dataset, concatenate_datasets
import json

def download_ag_news_train_dataset():
    dataset = load_dataset('ag_news')
    combined_dataset = concatenate_datasets([dataset['train']])

    category_files = {
        0: 'ag_news_world_train.jsonl',   # World
        1: 'ag_news_sports_train.jsonl',  # Sports
        2: 'ag_news_business_train.jsonl',# Business
        3: 'ag_news_tech_train.jsonl'     # Tech
    }

    for label, filename in category_files.items():
        category_dataset = combined_dataset.filter(lambda example: example['label'] == label)
        file_path = 'origin/' + filename
        
        with open(file_path, 'w', encoding='utf-8') as outfile:
            for idx, example in enumerate(category_dataset):
                text_parts = example['text'].split('\n', 1)
                record = {
                    "id": idx,
                    "content": text_parts[0]
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Successfully saved {len(category_dataset)} records to {file_path}")
        
def download_ag_news_test_dataset():
    dataset = load_dataset('ag_news')
    combined_dataset = concatenate_datasets([dataset['test']])

    category_files = {
        0: 'ag_news_world_test.jsonl',   # World
        1: 'ag_news_sports_test.jsonl',  # Sports
        2: 'ag_news_business_test.jsonl',# Business
        3: 'ag_news_tech_test.jsonl'     # Tech
    }

    for label, filename in category_files.items():
        category_dataset = combined_dataset.filter(lambda example: example['label'] == label)
        file_path = 'origin/' + filename
        
        with open(file_path, 'w', encoding='utf-8') as outfile:
            for idx, example in enumerate(category_dataset):
                text_parts = example['text'].split('\n', 1)
                record = {
                    "id": idx,
                    "content": text_parts[0],
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Successfully saved {len(category_dataset)} records to {file_path}")

if __name__ == '__main__':
    download_ag_news_train_dataset()
    download_ag_news_test_dataset()