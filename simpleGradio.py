import gradio as gr
import pandas as pd
import tempfile
import csv
from http import HTTPStatus
import dashscope
from dashscope import Generation
import os

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  # Vincent's API key


def response(prompt, instruction=None):
    messages = [{'role': 'user', 'content': prompt}]
    if instruction is not None:  # 如果提供了指令，则添加到messages中
        messages.insert(0, {'role': 'system', 'content': instruction})

    response = Generation.call(model='qwen-plus',
                               messages=messages,
                               seed=1234,
                               result_format='message',
                               stream=False,
                               incremental_output=False,
                               temperature=1.8,
                               top_p=0.9,
                               top_k=999
                               )
    if response.status_code == HTTPStatus.OK:
        message = response.output.choices[0]['message']['content']
        return message
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        return f"Error: Could not generate response with Status code: {response.status_code}, error code: {response.code}"


with gr.Blocks() as demo:
    input_text = gr.Textbox(label="输入文本")
    output_text = gr.Textbox(label="输出文本")
    submit_button = gr.Button("submit")
    # 使用 gr.ClearButton 来清除指定的输出组件
    clear_button = gr.ClearButton(components=[output_text], value="Clear processed data")

    # 当输入文本发生变化时，调用 response 函数并将结果显示在输出文本框中
    submit_button.click(fn=response, inputs=input_text, outputs=output_text)

demo.launch()

