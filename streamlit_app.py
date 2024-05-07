# ------------------------------------------
import os
import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
import datetime
import time
import random
import dashscope
from dashscope import Generation
from streamlit_option_menu import option_menu
from http import HTTPStatus
from PIL import Image


dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  # get api key from environment variable

st.set_page_config(layout="wide", page_title='TestGenius')

default_title = 'New Chat'
default_messages = [('user', 'What capabilities do you offer?'),
                    ('assistant', 'I excel at transforming uploaded images and texts into precise, multilingual test cases, streamlining the work of Business Analysts with efficiency and accuracy.')
                    ]

conversations = [{
    'id': 1,
    'title': 'Hello',
    'messages': default_messages
}]


def chat(user, message):
    with st.chat_message(user):
        print(user, ':', message)
        st.markdown(message)


if 'conversations' not in st.session_state:
    st.session_state.conversations = conversations
conversations = st.session_state.conversations

#  当前选择的对话
if 'index' not in st.session_state:
    st.session_state.index = 0

AVAILABLE_MODELS = [
    "qwen-max",
    "qwen-turbo",
]

with st.sidebar:
    st.image('logo.png')
    st.subheader('', divider='rainbow')
    st.write('')
    llm = st.selectbox('Choose your Model', AVAILABLE_MODELS, index=0)

    # if st.button('New Chat'):
    #     conversations.append({'title': default_title, 'messages': []})
    #     st.session_state.index = len(conversations) - 1

    titles = []
    for idx, conversation in enumerate(conversations):
        titles.append(conversation['title'])

    option = option_menu(
        'Conversations',
        titles,
        default_index=st.session_state.index
    )
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()

if uploaded_file:
    image_uploaded = Image.open(uploaded_file)
    image_path = image_uploaded.filename


def respond(prompt):
    messages = [
        {'role': 'user', 'content': prompt}]
    responses = Generation.call(model="qwen-turbo",
                                messages=messages,
                                result_format='message',  # 设置输出为'message'格式
                                stream=True,  # 设置输出方式为流式输出
                                incremental_output=True  # 增量式流式输出
                                )
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            yield response.output.choices[0]['message']['content'] + " "
        else:
            yield 'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            )


def respond_nonStream(prompt, instruction):
    messages = [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': prompt}]
    response = Generation.call(model="qwen-max",
                               messages=messages,
                               result_format='message',  # 设置输出为'message'格式
                               temperature=1,
                               top_p=0.8,
                               top_k=50)
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0]['message']['content']
    else:
        return 'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        )


def respond_image(prompt, image):
    messages = [
        {'role': 'user',
         "content": [
             {"image": image},
             {"text": prompt}
         ]
         }
    ]
    response = dashscope.MultiModalConversation.call(model="qwen-vl-max",
                                                     messages=messages,
                                                     result_format='message',  # 设置输出为'message'格式
                                                     )
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0]['message']['content']
    else:
        return 'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        )


prompt = st.chat_input("Enter your Questions")
st.session_state.messages = conversations[st.session_state.index]['messages']
if prompt:
    if conversations[st.session_state.index]['title'] == default_title:
        conversations[st.session_state.index]['title'] = prompt[:12]
    for user, message in st.session_state.messages:
        chat(user, message)
    chat('user', prompt)
    instruction = """
                # 角色定义
                您是一位多语言测试工程师AI，你能够首先判断用户输入的语言类型并告知用户，然后根据用户提供的产品功能描述，不仅生成详细准确的测试用例，
                而且确保测试用例的编写语言与用户输入内容的语言保持一致。无论用户用何种语言（支持的范围内）描述产品功能，您都能以相同语言输出测试用例。
                
                # 任务需求
                - 细致分析用户提交的一个或多个产品功能需求。
                - 根据需求文本或通过解读图片中的描述，生成语言一致的测试用例。
                
                # 输入处理
                - 用户输入可为不同语言的文字描述、图片，或二者的结合。
                - 对于多个需求，识别并以输入的语言和顺序生成相应的测试用例。
                
                # 格式和规范
                - 测试用例应以Markdown格式的表格显示，包含`用例编号`、`测试步骤`及`预期结果`三栏。
                - 与用户输入的语言保持一致，确保不发生语言混淆。
                - 若需求以图片呈现，须解释图像内容，并以相同语言形式处理文字需求。
                
                # 输出规则
                - 针对每项产品功能需求输出一个独立的Markdown表格式测试用例。
                - 若存在多个需求，依据需求的提交顺序生成并编号每个测试用例，保证输出的序列性和准确性。
                - 对无关产品功能的输入，以专业和恰当的语言回复：“作为语言适应型测试工程师AI，我专注于生成测试用例，请您提供具体的产品功能需求，以便我以相应语言回答。”
                
                # 示例1
                示例
                user:“当用户点击登录按钮时，如果已正确输入用户名和密码，系统应跳转到首页。”
                assistant: 
                您输入了中文的功能描述。
                | 用例编号 | 测试步骤                                               | 预期结果         |
                | -------- | ------------------------------------------------------ | ---------------- |
                | 1        | 1. 输入正确的用户名和密码 2. 点击登录按钮          | 系统跳转到首页   |
                
                # 示例2
                user: Clicking the registration button submits the data.
                assistant: 
                You entered requirement in ENGLISH.
                Test cases:
                | Case Number | Test Steps                                               | Expected Results         |
                | -------- | ------------------------------------------------------ | ---------------- |
                | 1        | 1. click the registration button        | data submitted   |
                """
    if uploaded_file:
        answer = respond_image(prompt, image_path)
    else:
        answer = respond_nonStream(prompt, instruction)
    st.session_state.messages.append(('user', prompt))
    st.session_state.messages.append(('assistant', answer))
    chat('assistant', answer)
else:
    for user, message in st.session_state.messages:
        chat(user, message)
