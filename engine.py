import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def carregar_cardapio(caminho_csv="cardapio_ifood_pronto.csv"):
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(
            f"O arquivo '{caminho_csv}' não foi encontrado. "
            "Execute 'python preparar_dados_kaggle.py' primeiro!"
        )
    return pd.read_csv(caminho_csv)

class MotorBuscaSemantica:
    def __init__(self, n_componentes_pca=16):
        self.modelo_ia = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.n_componentes = n_componentes_pca
        self.pca = PCA(n_components=n_componentes_pca, random_state=42)
        
    def indexar_cardapio(self, df):
        self.df = df.copy()
        textos = (
            self.df["prato"] + " (" + self.df["categoria"] + "): " + self.df["descricao"]
        ).tolist()
        
        self.embeddings_originais = self.modelo_ia.encode(textos, normalize_embeddings=True)
        
        max_k = min(len(self.df), self.embeddings_originais.shape[1])
        if self.n_componentes > max_k:
            self.n_componentes = max_k
            self.pca = PCA(n_components=self.n_componentes, random_state=42)
            
        projecao_pca = self.pca.fit_transform(self.embeddings_originais)
        normas = np.linalg.norm(projecao_pca, axis=1, keepdims=True)
        self.embeddings_pca = projecao_pca / np.clip(normas, 1e-12, None)
        
        self.variancia_explicada = float(np.sum(self.pca.explained_variance_ratio_) * 100)
        self.reducao_memoria = (1.0 - (self.embeddings_pca.nbytes / self.embeddings_originais.nbytes)) * 100

    def buscar(self, consulta_usuario, top_k=5):
        vetor_consulta_orig = self.modelo_ia.encode([consulta_usuario], normalize_embeddings=True)
        
        t0 = time.perf_counter()
        scores_orig = cosine_similarity(vetor_consulta_orig, self.embeddings_originais)[0]
        tempo_orig_ms = (time.perf_counter() - t0) * 1000
        
        t1 = time.perf_counter()
        vetor_consulta_pca = self.pca.transform(vetor_consulta_orig)
        norma_q = np.linalg.norm(vetor_consulta_pca, axis=1, keepdims=True)
        vetor_consulta_pca = vetor_consulta_pca / np.clip(norma_q, 1e-12, None)
        scores_pca = cosine_similarity(vetor_consulta_pca, self.embeddings_pca)[0]
        tempo_pca_ms = (time.perf_counter() - t1) * 1000
        
        resultados = self.df.copy()
        resultados["score_384d"] = np.round(scores_orig, 4)
        resultados["score_pca"] = np.round(scores_pca, 4)
        
        rank_orig = resultados.sort_values(by="score_384d", ascending=False).head(top_k)
        rank_pca = resultados.sort_values(by="score_pca", ascending=False).head(top_k)
        
        ids_orig = set(rank_orig["id"])
        ids_pca = set(rank_pca["id"])
        retencao_top_k = (len(ids_orig.intersection(ids_pca)) / top_k) * 100
        
        return {
            "ranking_original": rank_orig,
            "ranking_pca": rank_pca,
            "tempo_orig_ms": tempo_orig_ms,
            "tempo_pca_ms": tempo_pca_ms,
            "retencao_top_k": retencao_top_k
        }
