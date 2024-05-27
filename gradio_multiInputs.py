import gradio as gr
from http import HTTPStatus
import dashscope
from dashscope import Generation


dashscope.api_key = 'sk-73e9b0452a7e40048495d8ac8ab1afe4'  # Vincent's API key


def response(prompt, instruction):
    messages = [{'role': 'system', 'content': instruction},
                {'role': 'user', 'content': prompt}]

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


def process_prompts(prompts, instruction):
    """处理输入的prompts，调用模型，并返回结果。"""
    results = []
    for prompt in prompts.split("\n"):  # 分割多个prompts
        if prompt:  # 确保prompt不是空字符串
            output = response(prompt, instruction)
            results.append([prompt, output])
    return results


# 定义按钮点击后的事件处理函数，该函数会返回新的数据表格
def update_output(prompts, instruction):
    return process_prompts(prompts, instruction)


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("### 大模型测试工具")
        with gr.Accordion("输入说明"):
            gr.Markdown("请在下面的文本框中输入多个prompts，每个prompt占一行。")
            system = gr.Textbox(label="System prompts", lines=2,
                                value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
        prompts_input = gr.Textbox(label="Prompts", lines=5, placeholder="在这里输入prompts，每个一行...")
        submit_button = gr.Button("运行模型")
        output_table = gr.Dataframe(headers=["Prompt", "模型输出"])

        # 当按钮被点击时，调用update_output函数，并将返回的数据表格显示在output_table中
        submit_button.click(fn=update_output, inputs=[prompts_input, system], outputs=output_table)

    demo.launch()



