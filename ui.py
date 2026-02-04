"""
Gradio UI for YouTube RAG system.
"""

import gradio as gr
from ingest import ingest_video
from query import answer_question


def handle_ingest(youtube_url: str) -> str:
    if not youtube_url.strip():
        return "❌ Please provide a YouTube URL"
    try:
        result = ingest_video(youtube_url)
        return f"✅ Video {result['video_id']} ingested with {result['chunks']} chunks"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def handle_query(question: str) -> tuple:
    if not question.strip():
        return "❌ Please enter a question", []
    try:
        result = answer_question(question)
        timestamps = [f"{ts['start']} - {ts['end']}" for ts in result['timestamps']]
        return result['answer'], timestamps
    except Exception as e:
        return f"❌ Error: {str(e)}", []


with gr.Blocks(title="YouTube RAG") as app:
    gr.Markdown("# YouTube RAG Q&A")
    
    with gr.Tab("Ingest"):
        url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        ingest_btn = gr.Button("Ingest")
        ingest_output = gr.Textbox(label="Status", lines=3)
        ingest_btn.click(handle_ingest, url_input, ingest_output)
    
    with gr.Tab("Ask"):
        question_input = gr.Textbox(label="Question", lines=2)
        ask_btn = gr.Button("Ask")
        answer_output = gr.Textbox(label="Answer", lines=6)
        timestamps_output = gr.Dataframe(headers=["Timestamps"], label="These are the timestamps of the relevant transcript excerpts used to generate the answer.")
        ask_btn.click(handle_query, question_input, [answer_output, timestamps_output])


if __name__ == "__main__":
    app.launch()