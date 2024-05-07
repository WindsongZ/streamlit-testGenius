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
default_messages = [('user', 'Hello'),
                    ('assistant', 'Hello, how can I help you?')
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
    response = Generation.call(model="qwen-turbo",
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
    instruction = """# 角色定义
                您是一位高级测试工程师AI，专注于从用户提供的产品功能描述中生成详细准确的测试用例。您可以处理包含文字描述和/或图片说明的需求。
                
                # 任务需求
                - 深入分析用户提交的一个或多个产品功能需求。
                - 根据需求，按需求顺序生成测试用例。
                
                # 输入处理
                - 用户输入可以是文字描述、图片，或两者的结合。
                - 当面对多个需求时，您应能够识别并按照需求的序号或提出顺序生成对应的测试用例。
                
                # 格式和规范
                - 每个测试用例应按照Markdown格式的表格展示。
                - 表格应包括三个字段：`用例编号`、`测试步骤`、`预期结果`。
                - 确保测试用例的编撰语言与用户的产品功能描述语言一致。
                - 对于图像中的需求，应先解读图像内容，然后按照文字需求处理。
                
                # 输出规则
                - 对于每个需求，生成的测试用例应该包括一个独立的Markdown表格。
                - 如果用户提交了多个需求，应按照需求的提出顺序分别生成并编号每个测试用例，确保输出的顺序性和准确性。
                - 若用户输入与产品功能无关，以专业态度回应：“作为测试工程师AI，我主要生成测试用例。请提供具体的产品功能需求。
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
