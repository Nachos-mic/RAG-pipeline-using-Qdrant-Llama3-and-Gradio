import gradio as gr
from pipeline import rag_pipeline


def gradio_rag_interface(query):
    return rag_pipeline(query)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Demo RAG AI z Qdrant i Ollama")
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Wpisz pytanie", placeholder="Zadaj pytanie...")
            submit = gr.Button("Wyślij")
        with gr.Column():
            answer = gr.Textbox(label="Odpowiedź", lines=10)

    submit.click(fn=gradio_rag_interface, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch()
