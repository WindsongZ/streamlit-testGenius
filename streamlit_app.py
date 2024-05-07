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

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  # get api key from environment variable

st.set_page_config(layout="wide", page_title='TestGenius')

default_title = '新的对话'
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
    # st.image('assets/hero.png')
    st.subheader('', divider='rainbow')
    st.write('')
    llm = st.selectbox('Choose your Model', AVAILABLE_MODELS, index=1)

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


def respond(prompt):
    messages = [
        {'role': 'user', 'content': prompt}]
    responses = Generation.call(model=llm,
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
    messages = [{'role': 'system',
                 'content': instruction},
                {'role': 'user',
                 'content': prompt}
                ]
    response = Generation.call(model=llm,
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


prompt = st.chat_input("Enter your questions")
st.session_state.messages = conversations[st.session_state.index]['messages']
if prompt:
    if conversations[st.session_state.index]['title'] == default_title:
        conversations[st.session_state.index]['title'] = prompt[:12]
    for user, message in st.session_state.messages:
        chat(user, message)
    chat('user', prompt)
    system = 'You are a helpful assistant.'
    answer = respond_nonStream(prompt, system)
    st.session_state.messages.append(('user', prompt))
    st.session_state.messages.append(('assistant', answer))
    chat('assistant', answer)
else:
    for user, message in st.session_state.messages:
        chat(user, message)
