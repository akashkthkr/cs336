
"""
input_path: str Path to a text file with BPE tokenizer training data.

vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
initial byte vocabulary, vocabulary items produced from merging, and any special tokens).

special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
otherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
lary) to bytes (token bytes).

merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.
"""

import regex as re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count

NUM_THREADS = 2 * 10
CHUNK_SIZE = 128 * 1024  # 128KB

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def last_boundary(text: str) -> int:
    return max(text.rfind(' '), text.rfind('\n'), text.rfind('\t'))

def get_word_freq_in_batch_per_thread(input_file_path: str, start_idx: int, end_idx: int) -> dict[str, int]:
    ### Chunk size 
    word_freq = defaultdict(int)
    remainder = ""
    
    with open(input_file_path, "rb") as input_file:
        input_file.seek(start_idx)
        idx = start_idx
        
        while idx < end_idx:
            bytes_to_read = min(CHUNK_SIZE, end_idx - idx)
            chunk = input_file.read(bytes_to_read)
            
            usable_text = remainder + chunk.decode("utf-8", errors="ignore")
            last_boundary_idx = last_boundary(usable_text)
            
            if last_boundary_idx != CHUNK_SIZE:
                remainder = usable_text[last_boundary_idx + 1:]
                usable_text = usable_text[:last_boundary_idx + 1]
            else:
                remainder = ""
                
            pre_token_iterator = re.finditer(PAT, usable_text)            
            for token in pre_token_iterator:
                word_freq[token.group()] += 1
            
            if len(chunk) < CHUNK_SIZE:
                break
            idx += CHUNK_SIZE

    return word_freq

def compute_bpe_merges(word_freq_pre_tokens: dict[str, int], vocab_dict: dict[int, bytes], num_merges: int) -> list[tuple[bytes, bytes]]:
    merges:list[tuple[bytes, bytes]] = []
    word_bytes_list = defaultdict(list)
    for word in word_freq_pre_tokens.keys():
        word_bytes_list[word] = list(word.encode("utf-8"))
        
    for _ in range(num_merges):
        max_pair = None
        byte_pairs_freq = defaultdict(int)   
        for word, freq in word_freq_pre_tokens.items():
            temp_word = word_bytes_list[word]
                    
            for i in range(len(temp_word) - 1):
                byte_pairs_freq[(temp_word[i], temp_word[i+1])] += freq
                
        sorted_byte_pairs_freq = sorted(byte_pairs_freq.items(), key=lambda x: (x[1],x[0]), reverse=True)
        
        max_pair = sorted_byte_pairs_freq[0][0]
        
        first_bytes = vocab_dict[max_pair[0]]
        second_bytes = vocab_dict[max_pair[1]]
        merged_bytes = first_bytes + second_bytes
        
        
        merges.append((first_bytes, second_bytes))
        new_token_id = len(vocab_dict)
        vocab_dict[new_token_id] = merged_bytes
        
        for word in word_bytes_list:
            bytes_list = word_bytes_list[word]
            new_bytes_list = []
            i = 0
            while i < len(bytes_list):
                if i < len(bytes_list) - 1 and (bytes_list[i], bytes_list[i+1]) == max_pair:
                    new_bytes_list.append(new_token_id)
                    i += 2
                else:
                    new_bytes_list.append(bytes_list[i])
                    i += 1
            word_bytes_list[word] = new_bytes_list
    return merges


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # raise NotImplementedError
    if (vocab_size <= 256 + len(special_tokens)):
        raise ValueError("Vocab size is too small to fit the initial byte vocabulary and special tokens")
    
    
    vocab_dict = defaultdict(bytes) 
    word_freq_pre_tokens = defaultdict(int)
    for i in range(256):
        vocab_dict[i] = bytes([i])
        
        
    for sp_token in special_tokens:
        vocab_dict[len(vocab_dict)] = sp_token.encode("utf-8")
    input_chunking_file = open(input_path, "rb")
    boundaries = find_chunk_boundaries(input_chunking_file, NUM_THREADS, b"<|endoftext|>")
    input_chunking_file.close()
    
    with Pool(processes=NUM_THREADS) as pool:
        results = pool.starmap(get_word_freq_in_batch_per_thread, [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])])
        
    for result in results:
        for word, count in result.items():
            word_freq_pre_tokens[word] += count
            
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = compute_bpe_merges(word_freq_pre_tokens, vocab_dict, num_merges)
    return vocab_dict, merges
