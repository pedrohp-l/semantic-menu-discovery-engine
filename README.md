# 🍔 Motor de Busca Semântica de Cardápios (GenAI + PCA)

Este projeto é um MVP (Minimum Viable Product) focado na **Engenharia de IA e Machine Learning**, simulando a lógica contextual de recomendação do *Large Commerce Model (LCM)* do ecossistema iFood.

## 🎯 O Problema
Buscas tradicionais por palavras-chave falham quando o usuário digita intenções complexas (ex: *"jantar reconfortante para dia de chuva sem carne vermelha"*). Modelos de IA Generativa resolvem isso convertendo textos em vetores matemáticos (Embeddings), mas calcular a distância de milhares de vetores em alta dimensionalidade (384 dimensões) consome muita memória RAM e prejudica a latência em larga escala.

## 🚀 A Solução e Diferencial Técnico
1. **Sentence-Transformers:** Gera embeddings multilíngues a partir das descrições dos pratos e intenções do usuário.
2. **PCA (Análise de Componentes Principais):** Comprime os embeddings de 384 para 16 dimensões, reduzindo o consumo de memória vetorial em mais de 90%.
3. **Hybrid Engine:** Preserva a similaridade de cosseno geométrica (Recall) e mantém a ordem do ranking fiel ao espaço vetorial original, porém acelerando o tempo de busca da recomendação final.

## 🛠️ Tecnologias Utilizadas
* **Python & Pandas:** Limpeza, amostragem e manipulação do dataset.
* **Scikit-Learn:** Implementação do modelo matemático PCA.
* **Hugging Face (Sentence-Transformers):** Geração de embeddings semânticos densos.
* **Streamlit:** Interface visual para simulação interativa da busca.

## ⚙️ Como Executar
1. Instale as dependências: `pip install -r requirements.txt`
2. Enriqueça o dataset bruto: `python preparar_dados.py`
3. Rode a aplicação gráfica: `streamlit run app.py`
