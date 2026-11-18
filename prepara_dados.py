import pandas as pd
import numpy as np

def gerar_dataset_enriquecido(caminho_csv_kaggle="ifood-restaurants-february-2021.csv", n_restaurantes=150):
    print("Lendo amostra leve do arquivo do Kaggle...")
    colunas_uteis = ["name", "category", "price_range", "rating", "delivery_time", "delivery_fee"]
    
    df_kaggle = pd.read_csv(caminho_csv_kaggle, usecols=colunas_uteis, nrows=3000)
    df_kaggle = df_kaggle.dropna(subset=["name", "category", "delivery_time"])
    df_kaggle = df_kaggle[df_kaggle["rating"] > 0].drop_duplicates(subset=["name"])
    
    df_amostra = df_kaggle.sample(n=min(n_restaurantes, len(df_kaggle)), random_state=42).reset_index(drop=True)

    catalogo_pratos = {
        "Lanches": [
            ("Hambúrguer Artesanal Smash Duplo Bacon", "Lanche suculento para matar a fome no fim de semana. Pão brioche, dois blends bovinos grelhados, queijo cheddar derretido, bacon crocante e maionese da casa.", 38.90),
            ("Sanduíche Natural de Frango com Ricota", "Lanche leve, rápido e saudável para tarde quente. Pão integral de grãos, frango desfiado temperado com ervas, creme de ricota light, cenoura e alface.", 24.50)
        ],
        "Brasileira": [
            ("Feijoada Completa Tradicional", "Almoço pesado e farto de sábado para dias frios. Feijão preto encorpado com carnes nobres, arroz branco, couve refogada no alho, farofa crocante e laranja.", 52.00),
            ("Filé de Tilápia Grelhada com Purê de Mandioquinha", "Refeição caseira leve, saudável e fácil de digerir para o jantar, sem carne vermelha e sem glúten. Peixe fresco grelhado no azeite, purê cremoso e legumes no vapor.", 36.90)
        ],
        "Japonesa": [
            ("Poke Havaiano de Salmão Fresco", "Comida japonesa leve, fria, refrescante e saudável para dias de calor, zero lactose. Cubos de salmão, arroz japonês, abacate, pepino sunomono, manga e gergelim.", 49.90),
            ("Lámen Tradicional Quente (Missô Lamen)", "Sopa oriental bem quente e reconfortante ideal para noites frias e dias de chuva. Caldo rico de missô, macarrão artesanal, lombo suíno, ovo marinado e cebolinha.", 44.00)
        ],
        "Saudável": [
            ("Bowl Vegano Proteico de Quinoa e Grão-de-Bico", "Almoço ou jantar fitness 100% vegetal, sem carne, sem glúten e zero lactose. Quinoa real, grão-de-bico assado com páprica, abóbora cabotiá, espinafre e molho tahine.", 39.50),
            ("Salada Caesar com Tiras de Frango Grelhado", "Refeição super leve e baixa caloria para dieta em dias quentes. Mix de alfaces crocantes, peito de frango grelhado em tiras, lascas de parmesão e croutons.", 34.00)
        ],
        "Pizza": [
            ("Pizza Napolitana Margherita de Fermentação Natural", "Jantar quente e reconfortante para compartilhar em família ou dias de chuva, sem carne vermelha. Massa longa fermentação, molho de tomate italiano, muçarela de búfala e manjericão.", 68.00),
            ("Pizza de Calabresa Artesanal Acebolada", "Clássica pizza salgada bem assada para noite de sexta-feira. Muçarela derretida, linguiça calabresa defumada fatiada, cebola roxa e orégano.", 59.90)
        ],
        "Açaí": [
            ("Tigela de Açaí Zero Açúcar com Whey e Morango", "Lanche pós-treino gelado, energético e refrescante para dias de sol e calor. Açaí puro batido sem açúcar, proteína isolada, morangos frescos e pasta de amendoim.", 29.00)
        ],
        "Italiana": [
            ("Risoto Cremoso de Funghi Secchi com Azeite Trufado", "Jantar romântico, quente, sofisticado e reconfortante para noites chuvosas, sem carne vermelha. Arroz arbóreo cremoso, cogumelos funghi, vinho branco e parmesão.", 64.00),
            ("Nhoque de Mandioquinha ao Molho Bolonhesa", "Massa artesanal macia e bem quente com molho rico de tomate e carne bovina selecionada. Ideal para almoço de domingo ou dias frios.", 48.00)
        ],
        "Doces & Bolos": [
            ("Bolo Quente de Chocolate Belga com Brigadeiro", "Sobremesa doce, quente e indulgente para acompanhar um café em tarde chuvosa. Massa úmida de cacau com cobertura generosa de chocolate meio amargo.", 22.50)
        ]
    }

    pratos, descricoes, precos = [], [], []
    chaves_padrao = list(catalogo_pratos.keys())

    for idx, row in df_amostra.iterrows():
        cat_original = str(row["category"]).strip()
        opcoes = catalogo_pratos.get(cat_original, catalogo_pratos[chaves_padrao[idx % len(chaves_padrao)]])
        escolhido = opcoes[idx % len(opcoes)]
        
        tempo = int(row["delivery_time"])
        nota = round(float(row["rating"]), 1)
        
        pratos.append(f"{escolhido[0]}")
        descricoes.append(
            f"{escolhido[1]} Categoria da loja: {cat_original}. "
            f"Avaliação dos clientes: {nota} estrelas. Tempo estimado de entrega: {tempo} minutos."
        )
        precos.append(escolhido[2])

    df_final = pd.DataFrame({
        "id": range(1, len(df_amostra) + 1),
        "prato": pratos,
        "restaurante": df_amostra["name"].values,
        "categoria": df_amostra["category"].values,
        "nota": np.round(df_amostra["rating"].values, 1),
        "tempo_entrega_min": df_amostra["delivery_time"].astype(int).values,
        "taxa_entrega": np.round(df_amostra["delivery_fee"].values, 2),
        "preco": precos,
        "descricao": descricoes
    })

    df_final.to_csv("cardapio_ifood_pronto.csv", index=False)
    print(f"Sucesso! Arquivo 'cardapio_ifood_pronto.csv' gerado com {len(df_final)} itens reais.")

if __name__ == "__main__":
    gerar_dataset_enriquecido()
