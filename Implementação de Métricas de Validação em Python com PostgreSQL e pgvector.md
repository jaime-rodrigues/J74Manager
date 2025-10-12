# Implementação de Métricas de Validação em Python com PostgreSQL e pgvector

Este guia detalha como implementar métricas de validação para quantificar a precisão e validar melhorias em seu sistema de busca por similaridade de imagens, utilizando Python, PostgreSQL e a extensão pgvector.

## 1. Preparação do Ambiente

Certifique-se de que seu ambiente Python tenha as seguintes bibliotecas instaladas:

```bash
pip install psycopg2-binary scikit-learn transformers torch torchvision matplotlib seaborn numpy pandas umap-learn
```

*   `psycopg2-binary`: Para interagir com o PostgreSQL.
*   `scikit-learn`: Para cálculo de métricas como `precision_score` e `recall_score`.
*   `transformers` e `torch`/`torchvision`: Para carregar e usar o modelo CLIP.
*   `matplotlib`, `seaborn`, `numpy`, `pandas`: Para manipulação de dados e visualização.
*   `umap-learn`: Para redução de dimensionalidade (alternativa ao t-SNE).

## 2. Criação do Dataset de Teste (Ground Truth)

Um dataset de teste com *ground truth* é essencial. Ele consiste em:

*   **Imagens de Consulta**: Um conjunto de imagens que você usará para realizar as buscas.
*   **Resultados Esperados (Ground Truth)**: Para cada imagem de consulta, uma lista de IDs de imagens no seu banco de dados que são consideradas relevantes (incluindo as idênticas ou com pequenas alterações que deveriam ser encontradas).

**Exemplo de Estrutura do Ground Truth (Python Dictionary)**:

```python
ground_truth = {
    "query_image_1.jpg": ["db_image_id_A", "db_image_id_B", "db_image_id_C"], # Imagens relevantes para query_image_1
    "query_image_2.jpg": ["db_image_id_D", "db_image_id_E"],
    # ... e assim por diante
}
```

Para imagens idênticas ou com pequenas alterações, certifique-se de que o ID da imagem original e o ID da imagem alterada estejam ambos na lista de resultados esperados para a consulta da imagem original (e vice-versa, se a imagem alterada também for uma consulta).

## 3. Geração de Embeddings de Consulta

Utilize o mesmo modelo CLIP e o mesmo pipeline de pré-processamento que você usa para gerar os embeddings no seu banco de dados.

```python
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

def generate_clip_embedding(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs.pixel_values)
    return image_features.squeeze().cpu().numpy() # Retorna um array numpy

# Carregar o modelo e o processador CLIP
model_name = "openai/clip-vit-base-patch32" # Ou o modelo CLIP que você está usando
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Exemplo de geração de embedding para uma imagem de consulta
# query_embedding = generate_clip_embedding("path/to/query_image.jpg", model, processor)
```

## 4. Execução de Consultas no pgvector

Para cada imagem de consulta, execute uma busca por similaridade no seu banco de dados PostgreSQL. Lembre-se de configurar `hnsw.ef_search` para um valor adequado antes de cada consulta para otimizar a precisão.

```python
import psycopg2
import numpy as np

def search_similar_images(query_embedding, top_k, db_config, ef_search_param=100):
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # Definir ef_search para a sessão atual (ou globalmente se preferir)
        cur.execute(f"SET hnsw.ef_search = {ef_search_param};")

        # Converter o embedding para o formato de string do pgvector
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Consulta SQL para buscar vizinhos mais próximos
        query = f"""
            SELECT id, original_filename, filepath_reference, 1 - (embedding <-> %s::vector) AS similarity
            FROM image_embeddings
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """
        cur.execute(query, (embedding_str, embedding_str, top_k))
        results = cur.fetchall()
        cur.close()
        return results
    except (Exception, psycopg2.Error) as error:
        print(f"Erro ao conectar ou consultar o banco de dados: {error}")
        return []
    finally:
        if conn:
            conn.close()

# Exemplo de configuração do banco de dados
db_config = {
    "dbname": "your_db",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

# Exemplo de uso:
# query_embedding = generate_clip_embedding("path/to/query_image.jpg", model, processor)
# search_results = search_similar_images(query_embedding, top_k=5, db_config=db_config, ef_search_param=150)
# print(search_results)
```

## 5. Cálculo das Métricas de Avaliação

Agora, vamos implementar as funções para calcular `Precision@k`, `Recall@k` e `mAP`.

```python
from sklearn.metrics import precision_score, recall_score, average_precision_score

def calculate_metrics(ground_truth, all_query_results, k_values=[1, 5, 10]):
    all_precision_at_k = {k: [] for k in k_values}
    all_recall_at_k = {k: [] for k in k_values}
    all_ap = [] # Average Precision for mAP

    for query_image_name, expected_relevant_ids in ground_truth.items():
        retrieved_ids = [res[0] for res in all_query_results.get(query_image_name, [])] # Assume res[0] é o ID da imagem

        # Para cada k
        for k in k_values:
            retrieved_at_k = retrieved_ids[:k]
            
            # True positives para Precision@k e Recall@k
            true_positives_at_k = [1 if item in expected_relevant_ids else 0 for item in retrieved_at_k]
            
            # Precision@k
            if len(retrieved_at_k) > 0:
                precision_k = sum(true_positives_at_k) / len(retrieved_at_k)
                all_precision_at_k[k].append(precision_k)
            else:
                all_precision_at_k[k].append(0) # Nenhuma imagem retornada

            # Recall@k
            if len(expected_relevant_ids) > 0:
                recall_k = sum(true_positives_at_k) / len(expected_relevant_ids)
                all_recall_at_k[k].append(recall_k)
            else:
                all_recall_at_k[k].append(1) # Não há itens relevantes esperados, então recall é 1 se nada for retornado

        # Cálculo do Average Precision (para mAP)
        # Para average_precision_score, precisamos de um array binário de relevância para todos os itens possíveis
        # Isso pode ser complexo se o conjunto total de itens for muito grande. 
        # Uma abordagem simplificada é considerar apenas os itens retornados e os esperados.
        # Para uma implementação mais robusta de AP, você precisaria de um ranking completo ou um conjunto maior de resultados.
        
        # Simplificação para AP: cria um array binário de relevância para os itens recuperados
        y_true = [1 if item in expected_relevant_ids else 0 for item in retrieved_ids]
        y_scores = [1.0 - res[3] for res in all_query_results.get(query_image_name, [])] # 1 - similarity para distância

        if len(y_true) > 0 and len(set(y_true)) > 1: # average_precision_score requer pelo menos duas classes
            ap = average_precision_score(y_true, y_scores)
            all_ap.append(ap)
        elif len(y_true) > 0 and 1 in y_true: # Se só há relevantes e foram recuperados
            all_ap.append(1.0)
        else:
            all_ap.append(0.0)

    # Calcular a média das métricas
    mean_precision_at_k = {k: np.mean(v) for k, v in all_precision_at_k.items()}
    mean_recall_at_k = {k: np.mean(v) for k, v in all_recall_at_k.items()}
    mean_average_precision = np.mean(all_ap) if all_ap else 0.0

    return mean_precision_at_k, mean_recall_at_k, mean_average_precision

# Exemplo de como usar as funções:
# all_query_results = {}
# for query_img_path in ground_truth.keys():
#     query_embedding = generate_clip_embedding(query_img_path, model, processor)
#     results = search_similar_images(query_embedding, top_k=10, db_config=db_config, ef_search_param=150)
#     all_query_results[query_img_path] = results

# mean_prec, mean_rec, mAP = calculate_metrics(ground_truth, all_query_results)
# print(f"Mean Precision@k: {mean_prec}")
# print(f"Mean Recall@k: {mean_rec}")
# print(f"mAP: {mAP}")
```

## 6. Análise Qualitativa e Visualização de Embeddings

Para entender o porquê da baixa precisão, a visualização dos embeddings é uma ferramenta poderosa.

### 6.1. Redução de Dimensionalidade com UMAP ou t-SNE

```python
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_embeddings(embeddings, labels, title="Visualização de Embeddings", method='umap'):
    if method == 'umap':
        reducer = umap.UMAP(random_state=42)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Método de redução de dimensionalidade inválido. Use 'umap' ou 'tsne'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=labels, # Use labels para colorir pontos de imagens similares
        palette=sns.color_palette("hsv", len(np.unique(labels))),
        legend="full",
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.show()

# Exemplo de uso:
# Suponha que você tenha uma lista de embeddings e seus rótulos (e.g., ID da imagem original para agrupamento)
# all_embeddings = [emb1, emb2, ...]
# all_labels = [label1, label2, ...]
# visualize_embeddings(np.array(all_embeddings), all_labels, method='umap')
```

**Interpretação da Visualização**:

*   **Clusters Coesos**: Imagens idênticas ou muito similares devem formar clusters apertados no espaço 2D/3D. Se elas estiverem dispersas, isso indica que o modelo de embedding não está capturando a similaridade esperada.
*   **Separação de Classes**: Diferentes classes de imagens devem estar bem separadas. Se houver sobreposição significativa entre classes distintas, o modelo pode estar generalizando demais ou não distinguindo características importantes.

### 6.2. Inspeção Direta de Embeddings

Para diagnosticar problemas com imagens específicas, calcule a similaridade de cosseno diretamente entre os embeddings de pares de imagens que você espera que sejam similares, mas que o sistema não está retornando corretamente.

```python
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# Exemplo:
# emb_original = generate_clip_embedding("path/to/original.jpg", model, processor)
# emb_altered = generate_clip_embedding("path/to/altered.jpg", model, processor)
# similarity = calculate_cosine_similarity(emb_original, emb_altered)
# print(f"Similaridade entre original e alterada: {similarity}")
```

*   **Diagnóstico**: Se a similaridade for baixa (e.g., < 0.95 para imagens que deveriam ser quase idênticas), o problema está na **geração do embedding** (pré-processamento ou modelo). Se a similaridade for alta, mas a busca no pgvector falha, o problema é a **indexação HNSW**.

## 7. Validação de Melhorias

Após implementar qualquer alteração (ajuste de parâmetros HNSW, fine-tuning do modelo, otimização do pré-processamento), repita o processo de:

1.  **Geração de Embeddings de Consulta** (se o modelo ou pré-processamento mudou).
2.  **Execução de Consultas no pgvector** (com os novos parâmetros HNSW, se aplicável).
3.  **Cálculo das Métricas de Avaliação**.

Compare os novos valores de `Precision@k`, `Recall@k` e `mAP` com os valores de referência obtidos antes das mudanças. Uma melhoria consistente nessas métricas indicará que suas alterações foram eficazes.

## Conclusão

Ao seguir este guia, você poderá estabelecer um pipeline robusto para quantificar a precisão do seu sistema de busca por similaridade de imagens, diagnosticar as causas da baixa precisão e validar as melhorias de forma sistemática. A chave é a criação de um *ground truth* representativo e a análise cuidadosa tanto das métricas quantitativas quanto das visualizações qualitativas dos embeddings.

## Referências

As referências utilizadas para este guia são as mesmas do relatório anterior, que podem ser consultadas para aprofundamento nos tópicos abordados.
