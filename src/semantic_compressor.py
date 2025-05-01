import re
import torch
import spacy
import numpy as np
from typing import List, Tuple

class SemanticCompressor:
    def __init__(self, model, tokenizer, device, ave_info=0, eta=0):
        """
        Initialize the SemanticCompressor with a pre-trained model and tokenizer.
        
        Args:
            model: A pre-trained causal language model from Hugging Face.
            tokenizer: A tokenizer corresponding to the model.
            device: The device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.info = None
        self.eta = eta
        self.ave_info = ave_info
        self.model.eval()
        self.nlp = spacy.load("en_core_web_md")

    def _calculate_self_information(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Calculate the self-information of each token in the input text.
        
        Args:
            text: The input text.
        
        Returns:
            token_ids: A list of token IDs.
            token_self_info: A list of self-information values for each token.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
            
            probs = torch.softmax(logits, dim=-1)  # Shape: (batch_size, sequence_length, vocab_size)
            self_info = -torch.log(probs)  # Shape: (batch_size, sequence_length, vocab_size)
            token_self_info = self_info.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, sequence_length)
        
        token_ids = input_ids.squeeze(0).tolist()  # Shape: (sequence_length,)
        token_self_info = token_self_info.squeeze(0).tolist()  # Shape: (sequence_length,)

        self.info = np.mean([x for x in token_self_info if x > 0 and np.isfinite(x)])
        
        return token_ids, token_self_info

    def _map_self_info_to_words(self, text: str, token_ids: List[int], token_self_info: List[float]) -> Tuple[List[str], List[float]]:
        """
        Map token-level self-information back to word-level self-information.
        
        Args:
            text: The original input text.
            token_ids: A list of token IDs.
            token_self_info: A list of self-information values for each token.
        
        Returns:
            words: A list of words in the original text.
            word_self_info: A list of average self-information values for each word.
        """
        words = re.findall(r'\s*\S+', text)
        word_self_info = []
        current_token_idx = 0
        
        if "deepseek" in self.model.config.name_or_path:
            current_token_idx += 1  # Skip the first token '<｜begin▁of▁sentence｜>' for deepseek model
        
        for word in words:
            # Tokenize the word and get its token IDs
            word_tokens = self.tokenizer.tokenize(word)
            word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            word_token_self_info = []
            
            # Collect self-information for each token in the word
            for token_id in word_token_ids:
                if current_token_idx < len(token_ids):
                    if token_id != token_ids[current_token_idx]:
                        # Handle token mismatch (e.g., due to tokenizer differences)
                        print("Token mismatch detected!")
                        # Output the mismatched segments
                        print(f"cuurent_token_idx: {current_token_idx}")
                        print(f"Mismatched word tokens: {token_ids[current_token_idx]}, Decoded: {self.tokenizer.decode(token_ids[current_token_idx])}")
                        print(f"Mismatched token IDs: {token_id}, Decoded: {self.tokenizer.decode(token_id)}")
                    
                    word_token_self_info.append(token_self_info[current_token_idx])
                    current_token_idx += 1
            
            # Calculate the sum self-information for the word
            if word_token_self_info:
                word_self_info.append(np.sum(word_token_self_info))
                # word_self_info.append(np.mean(word_token_self_info))
            else:
                word_self_info.append(np.inf)  # If no tokens match, set self-information to inf
        
        return words, word_self_info

    def compress_text(self, text: str, reduce_ratio: float = 0.35, use_unit_info: bool = False) -> Tuple[str, List[str]]:
        """
        Compress the input text by removing words with self-information below the threshold.
        
        Args:
            text: The input text.
            reduce_ratio: The ratio of content to be removed (e.g., 0.35 means 35% of content will be removed).
        
        Returns:
            compressed_text: The compressed text.
            masked_words: A list of words that were filtered out.
        """
        token_ids, token_self_info = self._calculate_self_information(text)
        if use_unit_info:
            words, word_self_info = self._map_self_info_to_words(text, token_ids, token_self_info)
                
            if self.ave_info != 0:
                reduce_ratio = reduce_ratio * (1 - self.eta * (self.info - self.ave_info) / self.ave_info)
                reduce_ratio = max(0, min(1, reduce_ratio))
                
            doc = self.nlp(text)
            entities = list(doc.ents)
            entity_texts = set()
            for ent in entities:
                cleaned_ent = re.sub(r'^[\W_]+|[\W_]+$', '', ent.text.strip())
                entity_texts.add(cleaned_ent)
                
            # Calculate the threshold based on reduce_ratio
            if word_self_info:
                threshold = np.nanpercentile(word_self_info, reduce_ratio * 100)
            else:
                threshold = 0
            
            # Filter out words with self-information below the threshold
            compressed_words = []
            masked_words = []
            for word, info in zip(words, word_self_info):
                cleaned_word = re.sub(r'^[\W_]+|[\W_]+$', '', word.strip())
                if cleaned_word in entity_texts:
                    info = np.inf
                
                if info >= threshold:
                    compressed_words.append(word)
                else:
                    compressed_words.append(' []')
                    masked_words.append(word)
            
            compressed_text = ''.join(compressed_words)
            return compressed_text, masked_words, self.info
            
        else:
            if token_self_info:
                threshold = np.nanpercentile(token_self_info, reduce_ratio * 100)
            else:
                threshold = 0
            
            compressed_tokens = []
            masked_tokens = []
            for token_id, info in zip(token_ids, token_self_info):
                if info >= threshold:
                    compressed_tokens.append(token_id)
                else:
                    masked_tokens.append(token_id)
            
            return np.array(compressed_tokens), np.array(masked_tokens), self.info
        
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    compressor = SemanticCompressor(model, tokenizer, device)
    test_text = "$17, 155Friday, 166, Tuesday. The quick brown fox jumps over the lazy dog."
    
    compressed_text, masked_words, info = compressor.compress_text(test_text, reduce_ratio=0.3, use_unit_info=True)
    print(f"Compressed text: {compressed_text}")