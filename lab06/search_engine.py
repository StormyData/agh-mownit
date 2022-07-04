#!/bin/python3
import json
import pickle

import numpy as np
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import scipy.sparse
import tqdm

class SearchEngine:
    def __init__(self, m_type: str = "SVD"):

        self.set_type(m_type)

        self.vectors_used = 500
        self.ps = PorterStemmer()
        with open("data/index_to_uuid.pickle", "rb") as f:
            index_to_uuid = pickle.load(f)
            self.n_documents = len(index_to_uuid)

        with open("data/enumerated_words.pickle", "rb") as f:
            self.enumerated_words = pickle.load(f)

        with open("data/checkpoint_dict.json", "r") as f:
            cd = json.load(f)
            cd: dict
            rev_cd = {v: k for k, v in cd.items()}
            self.index_to_url = [rev_cd[index_to_uuid[index]] for index in range(self.n_documents)]

    def search_string(self, string: str, k: int):
        words = list(map(lambda s: self.enumerated_words[s], filter(lambda s: s in self.enumerated_words,
                                                               map(lambda s: self.ps.stem(s, to_lowercase=True),
                                                                   word_tokenize(string)))))
        if len(words) == 0:
            return "unknown words"
        array = np.zeros(len(self.enumerated_words))
        for word in words:
            array[word] += 1
        return self._search_vector(array, k)

    def _search_vector(self, array: np.ndarray, k: int):
        array /= np.linalg.norm(array)
        if self.svd:
            u, s, vh, weights = self.matrix
            s[self.vectors_used:] = 0
            output_arr = np.abs((((array @ u) @ scipy.sparse.diags(s)) @ vh) @ scipy.sparse.diags(weights))
        else:
            output_arr = np.abs(array @ self.matrix)
        indexes = list(range(self.n_documents))
        indexes.sort(key=lambda x: output_arr[x], reverse=True)
        return [(self.index_to_url[index], output_arr[index]) for index in indexes[:k]]

    def set_type(self, type: str):
        match type.upper():
            case "SVD":
                with open("data/SVD_with_vectors.pickle", "rb") as f:
                    self.matrix = pickle.load(f)
                self.svd = True
            case "IDF":
                with open("data/NormalizedMatrix.pickle", "rb") as f:
                    self.matrix = pickle.load(f)
                    self.matrix: scipy.sparse.dok_matrix
                self.svd = False
            case "NOIDF":
                with open("data/NormalizedMatrixNoIDF.pickle", "rb") as f:
                    self.matrix = pickle.load(f)
                    self.matrix: scipy.sparse.dok_matrix
                self.svd = False
            case _:
                return "invalid type"
        return type

    def set_vectors_used(self, n: int):
        self.vectors_used = n

    def get_max_vectors(self):
        if self.svd:
            return len(self.matrix[1])
        return 0


def main():
    se = SearchEngine()
    print("zapytania poprzedzone są znakiem q, np. qriffraff\nzmiana trybu poprzedzona jest znakiem t, "
          "np. tIDF\nzmiana ilości wektorów porprzedzona jest znakiem v, np. v1000\nby zakończyć e")
    while True:
        str = input(">")
        if str[0] == "q":
            result = se.search_string(str[1:], 20)
            if isinstance(result, str.__class__):
                print("unknown word")
                continue
            for url, score in result:
                print(f"{score} {url}")
        elif str[0] == "v":
            se.vectors_used = int(str[1:])
        elif str[0] == "t":
            print(se.set_type(str[1:]))
        elif str[0] == "e":
            return


if __name__ == "__main__":
    main()