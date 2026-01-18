import gradio as gr

def test(img):
    return "API WORKS"

gr.Interface(
    fn=test,
    inputs=gr.Image(),
    outputs=gr.Textbox()
).launch()
