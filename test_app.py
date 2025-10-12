import gradio as gr
import requests
import io
import os
from PIL import Image

# --- CONFIGURAÇÃO ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_SEARCH_URL = f"{API_BASE_URL}/images/search-by-upload"
UPLOADS_DIR = "uploads"

def search_images(query_image):
    if query_image is None:
        return []

    img_byte_arr = io.BytesIO()
    query_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('query.jpg', img_byte_arr, 'image/jpeg')}
    
    print(f"Enviando requisição para a API em: {API_SEARCH_URL}")
    
    try:
        response = requests.post(API_SEARCH_URL, files=files, params={'top_k': 5})
        response.raise_for_status()

        results = response.json()
        similar_images = results.get("similar_images", [])

        if not similar_images:
            print("Nenhuma imagem similar encontrada.")
            return []

        # --- CÓDIGO MODIFICADO AQUI ---
        # Criar uma lista de tuplas (caminho_da_imagem, legenda) para a galeria
        gallery_data = []
        for img in similar_images:
            # Constrói o caminho completo para a imagem localmente
            full_path = os.path.join(UPLOADS_DIR, img["filepath"])
            
            # Formata a similaridade como uma porcentagem
            similarity_score = img['similarity'] * 100
            
            # Cria a legenda usando os dados do schema ImageResult
            caption = f"Similaridade: {similarity_score:.1f}%\nArquivo: {img['filename']}"
            
            gallery_data.append((full_path, caption))

        print(f"Resultados com legendas: {gallery_data}")
        return gallery_data

    except requests.exceptions.RequestException as e:
        print(f"Erro ao chamar a API: {e}")
        raise gr.Error(f"Falha ao conectar com a API. Verifique se ela está rodando. Detalhes: {e}")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")
        raise gr.Error(f"Ocorreu um erro: {e}")

with gr.Blocks() as demo:
    gr.Markdown("# Teste Visual da API de Busca de Imagens")
    gr.Markdown("Faça o upload de uma imagem para encontrar as mais similares no banco de dados.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Imagem de Busca")
            submit_button = gr.Button("Buscar Imagens Similares")
        
        with gr.Column():
            output_gallery = gr.Gallery(
                label="Resultados Similares", 
                show_label=True, 
                elem_id="gallery",
                columns=5, 
                rows=1, 
                object_fit="contain", 
                height="auto"
            )

    submit_button.click(fn=search_images, inputs=input_image, outputs=output_gallery)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
