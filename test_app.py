import gradio as gr
import requests
import io
import os
from PIL import Image
import numpy as np
import time

# Imports para visualização
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÃO ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_PROCESS_URL = f"{API_BASE_URL}/images/process-folder/"
API_SEARCH_URL = f"{API_BASE_URL}/images/search-by-upload"
API_METRICS_URL = f"{API_BASE_URL}/metrics/compare-embeddings"
API_GET_EMBEDDINGS_URL = f"{API_BASE_URL}/embeddings/get-by-filepaths"
API_DIAGNOSTICS_URL = f"{API_BASE_URL}/diagnostics/compare-embeddings"
UPLOADS_DIR = "uploads"

# --- FUNÇÕES DE LÓGICA ---

def start_processing(folder_name, use_augmentation):
    """Inicia a tarefa de processamento de pasta na API Celery."""
    if not folder_name:
        raise gr.Error("Por favor, forneça o nome da pasta.")
    
    data = {
        'folder': folder_name,
        'use_augmentation': use_augmentation
    }
    print(f"Iniciando tarefa de indexação com dados: {data}")
    try:
        response = requests.post(API_PROCESS_URL, data=data)
        response.raise_for_status()
        task_info = response.json()
        # Inicia o loop de atualização de status
        return task_info['task_id'], f"Tarefa enviada para a fila. ID: {task_info['task_id']}", gr.update(visible=True)
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Falha ao iniciar a tarefa: {e}")

def get_task_status(task_id):
    """Consulta o status de uma tarefa Celery na API."""
    if not task_id:
        return "Nenhuma tarefa em execução.", gr.update(visible=False)
    
    status_url = f"{API_BASE_URL}/tasks/{task_id}"
    try:
        response = requests.get(status_url)
        response.raise_for_status()
        task = response.json()
        
        status_report = f"Status: {task['status']}"
        if task.get('progress'):
            status_report += f" | Progresso: {task['progress']}"
        if task.get('result'):
            status_report += f" | Resultado: {task['result']}"
        if task.get('error'):
            status_report += f" | Erro: {task['error']}"
            
        is_done = task['status'] in ['SUCCESS', 'FAILURE']
        return status_report, gr.update(visible=not is_done)
        
    except requests.exceptions.RequestException as e:
        return f"Erro ao consultar status: {e}", gr.update(visible=True)

# ... (outras funções de lógica permanecem as mesmas) ...
def search_images(query_image, search_scope):
    if query_image is None: return [], None, None, None, gr.update(visible=False)
    img_byte_arr = io.BytesIO(); query_image.save(img_byte_arr, format='JPEG'); img_byte_arr = img_byte_arr.getvalue()
    files = {'file': ('query.jpg', img_byte_arr, 'image/jpeg')}
    scope_value = 'all' if search_scope == "Tudo (Originais + Variações)" else 'original_only'
    data = {'scope': scope_value}
    try:
        response = requests.post(API_SEARCH_URL, files=files, data=data, params={'top_k': 5}); response.raise_for_status()
        results = response.json(); similar_images = results.get("similar_images", [])
        if not similar_images: return [], query_image, results, None, gr.update(visible=False)
        gallery_data = []; filepath_choices = []
        for img in similar_images:
            full_path = os.path.join(UPLOADS_DIR, img["filepath"]); similarity_score = img['similarity'] * 100
            caption = f"Similaridade: {similarity_score:.1f}%\nArquivo: {img['filename']}"
            gallery_data.append((full_path, caption)); filepath_choices.append(img["filepath"])
        return gallery_data, query_image, results, None, gr.update(visible=True), gr.update(choices=filepath_choices, value=filepath_choices[0])
    except requests.exceptions.RequestException as e: raise gr.Error(f"Falha ao conectar com a API de busca. Detalhes: {e}")

def get_metrics(original_image_state, search_results_state, evt: gr.SelectData):
    if original_image_state is None or search_results_state is None: raise gr.Error("Faça uma busca primeiro.")
    selected_image_data = search_results_state['similar_images'][evt.index]; selected_filepath = selected_image_data['filepath']
    original_img_bytes = io.BytesIO(); original_image_state.save(original_img_bytes, format='JPEG'); original_img_bytes = original_img_bytes.getvalue()
    files = {'query_image': ('original.jpg', original_img_bytes, 'image/jpeg')}; data = {'selected_filepath': selected_filepath}
    try:
        response = requests.post(API_METRICS_URL, files=files, data=data); response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e: raise gr.Error(f"Falha ao conectar com a API de métricas. Detalhes: {e}")

def visualize_results(search_results_state):
    if not search_results_state or not search_results_state.get('similar_images'): raise gr.Error("Realize uma busca com resultados antes.")
    all_embeddings = []; all_labels = []
    query_embedding = search_results_state.get('query_embedding')
    if query_embedding: all_embeddings.append(query_embedding); all_labels.append("Imagem de Busca")
    result_filepaths = [img['filepath'] for img in search_results_state['similar_images']]
    try:
        response = requests.post(API_GET_EMBEDDINGS_URL, data={'filepaths': result_filepaths}); response.raise_for_status()
        embeddings_map = response.json()
    except requests.exceptions.RequestException as e: raise gr.Error(f"Falha ao buscar embeddings da API. Detalhes: {e}")
    for i, path in enumerate(result_filepaths):
        if embeddings_map.get(path): all_embeddings.append(embeddings_map[path]); all_labels.append(f"Resultado {i+1}")
    if len(all_embeddings) < 2: raise gr.Error("Não há embeddings suficientes para a visualização.")
    reducer = umap.UMAP(random_state=42, n_neighbors=min(len(all_embeddings)-1, 15), min_dist=0.1)
    reduced_embeddings = reducer.fit_transform(np.array(all_embeddings))
    plt.figure(figsize=(10, 8)); sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=all_labels, palette=sns.color_palette("hsv", len(np.unique(all_labels))), legend="full", alpha=0.9, s=150)
    plt.title("Visualização UMAP dos Embeddings da Busca"); plt.xlabel("Componente UMAP 1"); plt.ylabel("Componente UMAP 2"); plt.legend(title="Imagens"); plt.grid(True)
    return plt.gcf()

def run_diagnostics(diagnostic_image, db_filepath):
    if diagnostic_image is None or not db_filepath: raise gr.Error("Forneça uma imagem e um filepath.")
    img_byte_arr = io.BytesIO(); diagnostic_image.save(img_byte_arr, format='JPEG'); img_byte_arr = img_byte_arr.getvalue()
    files = {'upload_image': ('diagnostic.jpg', img_byte_arr, 'image/jpeg')}; data = {'db_filepath': db_filepath}
    try:
        response = requests.post(API_DIAGNOSTICS_URL, files=files, data=data); response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e: raise gr.Error(f"Falha ao conectar com a API de diagnóstico. Detalhes: {e}")

# --- INTERFACE GRADIO ---
with gr.Blocks() as demo:
    gr.Markdown("# Ferramenta de Análise de Busca de Imagens")

    original_image_state = gr.State()
    search_results_state = gr.State()
    task_id_state = gr.State()

    with gr.Tabs():
        with gr.TabItem("Indexação de Imagens"):
            gr.Markdown("Use esta aba para iniciar o processo de indexação de imagens no banco de dados. O status da tarefa será atualizado automaticamente.")
            with gr.Row():
                folder_name_input = gr.Textbox(label="Nome da Pasta em 'uploads/'", placeholder="ex: imagens/produtos")
                use_augmentation_checkbox = gr.Checkbox(label="Usar Data Augmentation", value=False)
            start_button = gr.Button("Iniciar Indexação")
            
            with gr.Row():
                task_status_output = gr.Textbox(label="Status da Tarefa", interactive=False)
            
            # O loop de atualização agora é gerenciado pelo Gradio
            demo.load(get_task_status, inputs=task_id_state, outputs=[task_status_output], every=3)

        with gr.TabItem("Busca e Métricas"):
            # ... (código da aba de busca) ...
            gr.Markdown("**Passo 1:** Escolha o escopo da busca. **Passo 2:** Faça o upload de uma imagem e clique em 'Buscar'.")
            with gr.Row():
                with gr.Column(scale=1):
                    search_scope_radio = gr.Radio(["Apenas Originais", "Tudo (Originais + Variações)"], label="Escopo da Busca", value="Apenas Originais")
                    input_image = gr.Image(type="pil", label="Imagem de Busca")
                    submit_button = gr.Button("Buscar Imagens Similares")
                with gr.Column(scale=2):
                    output_gallery = gr.Gallery(label="Resultados da Busca", show_label=True, elem_id="gallery", columns=5, rows=1, object_fit="contain", height="auto")
            with gr.Row():
                metrics_output = gr.JSON(label="Relatório de Métricas Semânticas")

        with gr.TabItem("Análise Visual de Embeddings (UMAP)"):
            # ... (código da aba de visualização) ...
            gr.Markdown("Clique no botão para gerar um gráfico UMAP dos embeddings da última busca.")
            visualize_button = gr.Button("Visualizar Embeddings da Última Busca", visible=False)
            umap_plot = gr.Plot(label="Gráfico UMAP")

        with gr.TabItem("Diagnóstico de Embedding"):
            # ... (código da aba de diagnóstico) ...
            gr.Markdown("Compare o embedding de uma imagem de upload com um embedding armazenado no DB.")
            with gr.Row():
                diagnostic_image_input = gr.Image(type="pil", label="Imagem para Upload")
                db_filepath_dropdown = gr.Dropdown(label="Selecione o Filepath da Imagem no DB", choices=[])
            diagnostic_button = gr.Button("Executar Diagnóstico")
            diagnostic_output = gr.JSON(label="Relatório de Diagnóstico")

    # Ações dos botões
    start_button.click(fn=start_processing, inputs=[folder_name_input, use_augmentation_checkbox], outputs=[task_id_state, task_status_output])
    submit_button.click(fn=search_images, inputs=[input_image, search_scope_radio], outputs=[output_gallery, original_image_state, search_results_state, metrics_output, visualize_button, db_filepath_dropdown])
    output_gallery.select(fn=get_metrics, inputs=[original_image_state, search_results_state], outputs=metrics_output)
    visualize_button.click(fn=visualize_results, inputs=[search_results_state], outputs=[umap_plot])
    diagnostic_button.click(fn=run_diagnostics, inputs=[diagnostic_image_input, db_filepath_dropdown], outputs=[diagnostic_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
