import gradio as gr
import pandas as pd
# import tempfile
from http import HTTPStatus
import dashscope
from dashscope import Generation
import os
# import logging
from testAny import check_df_english, check_df_tags
import concurrent.futures
import datetime

# 设置日志记录
# logging.basicConfig(level=logging.INFO)

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  # Vincent's API key

def response(prompt, instruction=None):
    messages = [{'role': 'user', 'content': prompt}]
    if instruction is not None:  # 如果提供了指令，则添加到messages中
        messages.insert(0, {'role': 'system', 'content': instruction})

    try:
        response = Generation.call(model='qwen-plus',
                                   messages=messages,
                                   seed=1234,
                                   result_format='message',
                                   stream=False,
                                   incremental_output=False,
                                   temperature=0.85,
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
    except Exception as e:
        print(f"Failed to generate response: {e}")
        return f"Error: Failed to generate response due to an error."

def format_full_prompt(df, introduction):
    # 为每个 row 创建 context，拼接RAG1和2
    df['context'] = df.apply(lambda row: f"{row['RAG1']}-{row['RAG2']}", axis=1)

    # 准备用于 format 的字典
    column_list = df.drop('full_prompt', axis=1).columns.tolist()  # 去除full_prompt列，其他的都为参数
    format_dict = df[column_list].apply(lambda x: dict(zip(x.index, x)), axis=1)
    if len(introduction) >= 200:
        df['full_prompt'] = introduction
    # 使用 apply() 和 lambda 函数格式化 full_prompt 列
    df['full_prompt'] = df.apply(lambda row: row['full_prompt'].format(**format_dict[row.name]), axis=1)

    # 删除临时创建的 context 列
    df.drop(columns=['context'], inplace=True)
    return df

def process_xlsx(xlsx_file, instruction=None, loops=1):
    try:
        # 读取xlsx文件到pandas DataFrame
        df = pd.read_excel(xlsx_file)
        # 格式化prompts
        formatted_df = format_full_prompt(df, instruction)
        if loops > 1:
            df_list = [formatted_df.copy() for _ in range(loops - 1)]
            # 使用pd.concat一次性合并所有副本
            formatted_df = pd.concat([formatted_df] + df_list, ignore_index=True)

        # 调用response时，根据instruction是否为None自动处理
        formatted_df['Response'] = formatted_df['full_prompt'].apply(lambda prompt: response(prompt, instruction))

        # check df with tags and english
        formatted_df = check_df_tags(formatted_df)
        formatted_df = check_df_english(formatted_df)

        # 使用一个文件路径保存处理后的xlsx
        date_str = datetime.datetime.now().strftime("%m%d%H")
        times_str = str(20 * loops)
        output_path = 'output'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_name = f"{date_str}_{times_str}times_output.xlsx"
        file_path = os.path.join(output_path, file_name)
        formatted_df.to_excel(file_path, index=False, engine='openpyxl')
        return formatted_df, file_path
    except Exception as e:
        print(f"Failed to process xlsx: {e}")
        return None, None

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 大模型xlsx处理工具")
        with gr.Accordion("输入说明"):
            gr.Markdown("请上传一个xlsx文件，文件应包含prompts。")
            system_instruction = gr.Textbox(label="System Instruction", lines=2,
                                            value=" ")
            slider = gr.Slider(minimum=1, maximum=10, step=1, label="循环次数", value=1)

        file_input = gr.File(label="上传xlsx文件")
        submit_button = gr.Button("处理xlsx")

        output_table = gr.Dataframe(label="处理后的数据")
        output_file = gr.File(label="下载处理后的文件")
        clear_data = gr.ClearButton(components=[output_table, output_file], value="Clear processed data")
        clear_all = gr.ClearButton(components=[file_input, output_table, output_file], value="Clear console")
        def update_output(xlsx_file, instruction, loops):
            if xlsx_file is not None:
                formatted_df, tmp_path = process_xlsx(xlsx_file, instruction, loops)
                # 使用线程池并行处理每一行
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    formatted_df['Response'] = list(executor.map(response, formatted_df['full_prompt'], [instruction] * len(formatted_df)))

                if formatted_df is not None:
                    return formatted_df, tmp_path  # 返回DataFrame和文件路径

        submit_button.click(fn=update_output, inputs=[file_input, system_instruction, slider],
                            outputs=[output_table, output_file])

    demo.launch()


if __name__ == "__main__":
    main()
