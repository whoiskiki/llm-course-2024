import numpy as np
from minhash import MinHash
import collections

class MinHashLSH(MinHash):
    def __init__(self, num_permutations: int, num_buckets: int, threshold: float):
        self.num_permutations = num_permutations
        self.num_buckets = num_buckets
        self.threshold = threshold
        
    def get_buckets(self, minhash: np.array) -> np.array:
        '''
        Возвращает массив из бакетов, где каждый бакет представляет собой N строк матрицы сигнатур.
        '''
        minhash_buckets = []
        r = len(minhash) // self.num_buckets
        extra = len(minhash) % self.num_buckets

        start_index = 0
        for i in range(self.num_buckets):
            end_index = start_index + r + (1 if i < extra else 0)
            minhash_buckets.append(minhash[start_index:end_index])
            start_index = end_index

        return minhash_buckets

    def get_similar_candidates(self, buckets) -> list[tuple]:
        '''
        Находит потенциально похожих кандидатов.
        Кандидаты похожи, если полностью совпадают мин хеши хотя бы в одном из бакетов.
        Возвращает список из таплов индексов похожих документов.
        '''
        candidate_pairs = set()
        bucket_dict = collections.defaultdict(list)

        for doc, bucket in enumerate(buckets):
            for bucket_signature in bucket:
                bucket_dict[tuple(bucket_signature)].append(doc)

        print(bucket_dict)

        for doc_indices in bucket_dict.values():
            if len(doc_indices) > 1:
                print('here')
                for i in range(len(doc_indices)):
                    for j in range(i + 1, len(doc_indices)):
                        candidate_pairs.add((min(doc_indices[i], doc_indices[j]), max(doc_indices[i], doc_indices[j])))

        return list(candidate_pairs)
        
    def run_minhash_lsh(self, corpus_of_texts: list[str]) -> list[tuple]:
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        print(f'OCCURANCE MATRIX: {occurrence_matrix}')
        minhash = self.get_minhash(occurrence_matrix)
        print(f'MINHASH VALUES: {minhash}')
        buckets = self.get_buckets(minhash)
        print(f'BUCKETS: {buckets}')
        similar_candidates = self.get_similar_candidates(buckets)
        print(f'SIMILAR CANDIDATES: {similar_candidates}')
        
        return similar_candidates
    
