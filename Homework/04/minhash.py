import re
import pandas as pd
import numpy as np


class MinHash:
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def preprocess_text(self, text: str) -> str:
        return re.sub("( )+|(\n)+", " ", text).lower()

    def tokenize(self, text: str) -> set:
        text = self.preprocess_text(text)
        return set(text.split(' '))

    def get_occurrence_matrix(self, corpus_of_texts: list[set]) -> pd.DataFrame:
        '''
        Получение матрицы вхождения токенов. Строки - это токены, столбы это id документов.
        id документа - нумерация в списке начиная с нуля
        '''
        tokenized_corpus = [self.tokenize(str(text)) for text in corpus_of_texts]
        all_tokens = sorted(set(token for tokens in tokenized_corpus for token in tokens))

        token_dict = {token: [0] * len(tokenized_corpus) for token in all_tokens}

        for doc_idx, tokens in enumerate(tokenized_corpus):
            for token in tokens:
                token_dict[token][doc_idx] = 1

        df = pd.DataFrame.from_dict(token_dict, orient='index', columns=range(len(tokenized_corpus)))
        df.sort_index(inplace=True)

        print(df)

        return df

    def is_prime(self, a):
        if a % 2 == 0:
            return a == 2
        d = 3
        while d * d <= a and a % d != 0:
            d += 2
        return d * d > a

    def get_new_index(self, x: int, permutation_index: int, prime_num_rows: int) -> int:
        '''
        Получение перемешанного индекса.
        values_dict - нужен для совпадения результатов теста, а в общем случае используется рандом
        prime_num_rows - здесь важно, чтобы число было >= rows_number и было ближайшим простым числом к rows_number

        '''
        values_dict = {
            'a': [3, 4, 5, 7, 8],
            'b': [3, 4, 5, 7, 8]
        }
        a = values_dict['a'][permutation_index]
        b = values_dict['b'][permutation_index]
        return (a * (x + 1) + b) % prime_num_rows

    def get_minhash_similarity(self, array_a: np.array, array_b: np.array) -> float:
        '''
        Вовзращает сравнение minhash для НОВЫХ индексов. То есть: приходит 2 массива minhash:
            array_a = [1, 2, 1, 5, 3]
            array_b = [1, 3, 1, 4, 3]

            на выходе ожидаем количество совпадений/длину массива, для примера здесь:
            у нас 3 совпадения (1,1,3), ответ будет 3/5 = 0.6
        '''
        corr_answers = np.sum(array_a == array_b)/len(array_a)
        return corr_answers


    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        similar_pairs = []
        col = min_hash_matrix.shape[1]

        for i in range(col - 1):
            for j in range(i+1, col):
                sim = self.get_minhash_similarity(min_hash_matrix[:, i], min_hash_matrix[:, j])
                if sim > self.threshold:
                    similar_pairs.append((i, j))
        return similar_pairs


    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
        num_docs = min_hash_matrix.shape[1]
        similarity_matrix = np.zeros((num_docs, num_docs))

        for i in range(num_docs):
            for j in range(i, num_docs):
                similarity = np.sum(min_hash_matrix[:, i] == min_hash_matrix[:, j]) / min_hash_matrix.shape[0]
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Матрица симметрична

        return list(similarity_matrix)


    def get_next_prime(self, n):
        """Находит следующее простое число, большее или равное n."""
        if n <= 2:
            return 2
        while True:
            if self.is_prime(n):
                return n
            n += 1


    def get_minhash(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. MinHash содержит в себе новые индексы.

        new index = (2*(index +1) + 3) % 3

        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу
        [1, 0, 1]
        [1, 0, 1]
        [0, 1, 1]

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 0
        Doc2 : 2
        Doc3 : 0
        '''
        num_rows, num_cols = occurrence_matrix.shape
        prime_num_rows = self.get_next_prime(num_rows)
        minhash_matrix = np.full((self.num_permutations, num_cols), np.inf)

        # Генерация подписи для каждого документа
        for perm in range(self.num_permutations):
            for row_idx in range(num_rows):
                new_index = self.get_new_index(row_idx, perm, prime_num_rows)

                for doc_idx in range(num_cols):
                    if occurrence_matrix.iloc[row_idx, doc_idx] == 1:
                        minhash_matrix[perm, doc_idx] = min(float(minhash_matrix[perm, doc_idx]), new_index)

        print(minhash_matrix)

        return minhash_matrix

    def run_minhash(self, corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)

        return similar_pairs


class MinHashJaccard(MinHash):
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def get_jaccard_similarity(self, set_a: set, set_b: set) -> float:
        '''
        Вовзращает расстояние Жаккарда для двух сетов.
        '''
        return

    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        return

    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''

        return

    def get_minhash_jaccard(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. Но в качестве мин хеша выписываем минимальный исходный индекс, не новый.
        В такой ситуации можно будет пользоваться расстояние Жаккрада.

        new index = (2*(index +1) + 3) % 3

        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу
        [1, 0, 1] index: 2
        [1, 0, 1] index: 1
        [0, 1, 1] index: 0

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 2
        Doc2 : 0
        Doc3 : 2

        '''
        return

    def run_minhash(self, corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash_jaccard(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs


