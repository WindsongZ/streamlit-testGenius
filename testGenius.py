import random
from http import HTTPStatus
from dashscope import Generation
import dashscope
import gradio as gr

# dashscope.api_key = 'sk-73e9b0452a7e40048495d8ac8ab1afe4'  # Vincent's API key
dashscope.api_key = 'sk-83b8ed0ead0849ae9e63a2ae5bdbde0d'  # Rayman's API key


def respond_nonStream(prompt, chat_history, instruction, model):
    # 构建对话消息结构
    messages = [{'role': 'system', 'content': instruction},
                {'role': 'user', 'content': prompt}]
    print(f"Messages: {messages}")

    # 初始化空字符串以聚合响应
    full_response = ""

    # 调用AI模型生成响应
    try:
        response = Generation.call(model=model,
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


def respond(prompt, chat_history, instruction, model, if_stream='Stream'):
    if if_stream == 'Stream':
        messages = [{'role': 'system',
                     'content': instruction},
                    {'role': 'user',
                     'content': prompt}
                    ]
        full_response = ""  # 初始化空字符串以聚合响应
        # -------流式输出-------
        responses = Generation.call(model=model,
                                    messages=messages,
                                    # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                                    seed=1234,
                                    # 将输出设置为"message"格式
                                    result_format='message',
                                    stream=True,  # 设置输出方式为流式输出
                                    incremental_output=True,  # 增量式流式输出
                                    temperature=1.8,
                                    top_p=0.9,
                                    top_k=999)
        # 确保聊天历史至少有当前会话的开始
        if not chat_history or chat_history[-1][0] != prompt:
            chat_history.append((prompt, ""))
        # 循环处理每个流式响应
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                # 累加每次流式响应的内容
                text = response.output.choices[0]['message']['content']
                full_response += text
                # 更新聊天历史的最后一项
                last_turn = list(chat_history[-1])
                last_turn[1] = full_response
                chat_history[-1] = tuple(last_turn)
                yield "", chat_history  # 实时输出当前的聊天历史
            else:
                # 如果出错，构建错误信息并更新最后一项
                full_response = 'Request id: {}, Status code: {}, error code: {}, error message: {}'.format(
                    response.request_id, response.status_code,
                    response.code, response.message
                )
                last_turn = list(chat_history[-1])
                last_turn[1] = full_response
                chat_history[-1] = tuple(last_turn)
                yield "", chat_history
                break  # 出现错误时终止循环

    elif if_stream == 'Non-Stream':
        # 构建对话消息结构
        messages = [{'role': 'system', 'content': instruction},
                    {'role': 'user', 'content': prompt}]
        print(f"Messages: {messages}")

        # 调用AI模型生成响应
        try:
            response = Generation.call(model=model,
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



llm_model_list = ['qwen-turbo', 'qwen-plus', 'qwen-max']
init_llm = llm_model_list[0]

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # AI TestGenius
        A simple LLM app for generating test cases from function design.
        """)
    history = [["Hello", "Hello, how can I help you?"]]
    chatbot = gr.Chatbot(history)
    msg = gr.Textbox(label="Prompt")
    with gr.Accordion(label="Advanced options", open=False):
        system = gr.Textbox(label="System prompts", lines=2,
                            value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
        # 选择模型
        llm = gr.Dropdown(
            llm_model_list,
            label='Choose LLM Model',
            value=init_llm,
            interactive=True
        )
        # 选择是否流式输出
        if_stream = gr.Dropdown(
            ["Stream", "Non-Stream"],
            label='Choose Streaming',
            value="Stream",
            interactive=True
        )

    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
    btn.click(respond, inputs=[msg, chatbot, system, llm, if_stream], outputs=[msg, chatbot])  # click to submit
    msg.submit(respond, inputs=[msg, chatbot, system, llm, if_stream], outputs=[msg, chatbot])  # Press enter to submit

# 运行界面
if __name__ == "__main__":
    gr.close_all()
    demo.launch()
