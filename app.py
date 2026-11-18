import streamlit as st
from engine import carregar_cardapio, MotorBuscaSemantica

st.set_page_config(page_title="iFood GenAI - Semantic Discovery + PCA", layout="wide")

st.title("🍔 iFood Semantic Menu Discovery & Compressão Vetorial (PCA)")
st.markdown(
    "Motor de busca contextual inspirado no **Large Commerce Model (LCM)**, combinando dados reais "
    "de restaurantes do iFood (Kaggle) com **redução de dimensionalidade via PCA** para otimização de memória RAM e latência."
)

st.sidebar.header("⚙️ Parâmetros de Engenharia")
n_comp = st.sidebar.slider("Dimensões retidas no PCA (k)", min_value=2, max_value=64, value=16, step=2)
top_k = st.sidebar.slider("Quantidade de resultados (Top-K)", min_value=3, max_value=10, value=5)

@st.cache_resource
def inicializar_sistema(k):
    df = carregar_cardapio("cardapio_ifood_pronto.csv")
    motor = MotorBuscaSemantica(n_componentes_pca=k)
    motor.indexar_cardapio(df)
    return motor, len(df)

try:
    motor, total_itens = inicializar_sistema(n_comp)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Restaurantes Indexados", f"{total_itens}")
    col2.metric("Dimensão Original -> PCA", f"384D → {motor.n_componentes}D")
    col3.metric("Variância Explicada (PCA)", f"{motor.variancia_explicada:.1f}%")
    col4.metric("Economia de Memória Vetorial", f"{motor.reducao_memoria:.1f}%")

    st.divider()

    consulta = st.text_input(
        "🔍 Digite o que o cliente deseja comer (busca por intenção, clima ou restrição alimentar):",
        "Jantar quente e reconfortante para dia de chuva sem carne vermelha"
    )

    if consulta:
        res = motor.buscar(consulta, top_k=top_k)
        
        st.info(
            f"📊 **Fidelidade do Ranking (Recall@{top_k}):** O índice comprimido para **{motor.n_componentes} dimensões** "
            f"manteve **{res['retencao_top_k']:.0f}%** dos mesmos pratos recomendados pelo modelo completo de 384 dimensões."
        )

        col_esq, col_dir = st.columns(2)
        colunas_exibicao = ["prato", "restaurante", "categoria", "preco", "tempo_entrega_min", "nota"]

        with col_esq:
            st.subheader(f"🎯 Busca Original (384D) — {res['tempo_orig_ms']:.2f} ms")
            st.dataframe(
                res["ranking_original"][colunas_exibicao + ["score_384d"]],
                use_container_width=True,
                hide_index=True
            )

        with col_dir:
            st.subheader(f"⚡ Busca Comprimida PCA ({motor.n_componentes}D) — {res['tempo_pca_ms']:.2f} ms")
            st.dataframe(
                res["ranking_pca"][colunas_exibicao + ["score_pca"]],
                use_container_width=True,
                hide_index=True
            )

except FileNotFoundError as erro:
    st.error(str(erro))
