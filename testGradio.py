import gradio as gr
import random
import time
from http import HTTPStatus
import dashscope
from dashscope import Generation

dashscope.api_key = 'sk-73e9b0452a7e40048495d8ac8ab1afe4'  # Vincent's API key

with gr.Blocks() as demo:
    history = [["Hello","Hello, how can I help you?"]]
    chatbot = gr.Chatbot(history)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])


    def respond_nonStream(prompt, chat_history):
        # 构建对话消息结构
        messages = [{'role': 'system', 'content': "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."},
                    {'role': 'user', 'content': prompt}]
        print(f"Messages: {messages}")

        # 初始化空字符串以聚合响应
        full_response = ""

        # 调用AI模型生成响应
        try:
            response = Generation.call(model='qwen-turbo',
                                       messages=messages,
                                       seed=1234,
                                       result_format='message',
                                       stream=False,
                                       incremental_output=False)
            print(f"Response: {response}")

            if not chat_history or chat_history[-1][0] != prompt:
                chat_history.append([prompt, ""])
            print(f"old chat history: {chat_history}")
            if response.status_code == HTTPStatus.OK:
                # 获取响应中的消息内容
                message = response.output.choices[0]['message']['content']
                print(f"Generated message: {message}")

                # 更新聊天历史记录中的最后一条记录
                chat_history[-1] = [prompt, message]
                print(f"Updated chat_history: {chat_history}")
                return "", chat_history
            else:
                print(f"Error: Received response status {response.status_code}")
                return "Error: Could not generate response", chat_history

        except Exception as e:
            print(f"Exception occurred: {e}")
            return f"Exception occurred: {e}", chat_history

    msg.submit(respond_nonStream, inputs=[msg, chatbot], outputs=[msg, chatbot])

if __name__ == "__main__":
    demo.launch()