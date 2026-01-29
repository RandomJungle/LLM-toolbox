import os
from typing import Optional, List, Dict

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from mistralai import Mistral
from tqdm import tqdm

from converter import convert_content_to_json
from preprocessing import (
    chunk_dataframe,
    read_prompt_file,
)


def query_embeddings(
        dataframe: pd.DataFrame,
        model_name: Optional[str] = 'mistral-embed',
        num_chunks: Optional[int] = 1) -> pd.DataFrame:

    if not 'id' in dataframe.columns:
        dataframe['id'] = dataframe.index + 1

    data = dataframe[['id', 'text']]
    chunks = chunk_dataframe(data, num_chunks)
    outputs = []

    client = Mistral(
        api_key=os.getenv('MISTRAL_API_KEY', '')
    )
    for chunk in tqdm(chunks):
        chunk_response = client.embeddings.create(
            model=model_name,
            inputs=chunk['text'].tolist()
        )
        chunk[f'{model_name}_embeddings'] = [emb.embedding for emb in chunk_response.data]
        outputs.append(chunk)

    output_dataframe = pd.concat(outputs, ignore_index=True)
    merged = pd.merge(
        left=dataframe,
        right=output_dataframe[['id', f'{model_name}_embeddings']],
        on='id',
        how='left',
        validate='1:1'
    )
    return merged


def query_with_dataframe(
        dataframe: pd.DataFrame,
        prompt: Dict,
        model_name: Optional[str] = 'mistral-large-latest',
        temperature: Optional[float] = 0.2,
        num_chunks: Optional[int] = 1,
        stream: Optional[bool] = False) -> pd.DataFrame:

    data = dataframe[['id', 'text']]
    chunks = chunk_dataframe(data, num_chunks)
    responses = []

    client = Mistral(
        api_key=os.getenv('MISTRAL_API_KEY', '')
    )

    for chunk in tqdm(chunks):
        chunk_json = chunk.to_json(orient='records')
        messages = [
            {
                'content': prompt.get('role'),
                'role': 'system',
            },
            {
                'content': prompt.get('content') + f'<t>{chunk_json}</t>',
                'role': 'user',
            },
        ]
        if not stream:
            response = client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=temperature,
                response_format={
                    'type': 'json_object',
                }
            )
            content = response.choices[0].message.content
        else:
            response = client.chat.stream(
                model=model_name,
                messages=messages,
                temperature=temperature,
                response_format={
                    'type': 'json_object',
                }
            )
            collected_messages = []
            for res_chunk in response:
                chunk_content = res_chunk.data.choices[0].delta.content
                collected_messages.append(chunk_content)
                print(chunk_content)
            content = ''.join([m for m in collected_messages if m is not None])
        mini_df = pd.DataFrame.from_records(
            convert_content_to_json(content)
        )
        responses.append(mini_df)

    output_dataframe = pd.concat(responses, ignore_index=True)
    merged = pd.merge(
        left=dataframe,
        right=output_dataframe[prompt.get('output_columns')],
        on='id',
        how='left'
    )
    return merged


def query_with_documents(
        documents: Dict[str, str],
        prompt: Dict,
        output_path: str,
        model_name: Optional[str] = 'mistral-large-latest',
        temperature: Optional[float] = 0.2,
        stream: Optional[bool] = False) -> None:

    client = Mistral(
        api_key=os.getenv('MISTRAL_API_KEY', '')
    )

    for document_name, document in tqdm(documents.items()):

        messages = [
            {
                'content': prompt.get('role'),
                'role': 'system',
            },
            {
                'content': prompt.get('content') + f'<doc>{document}</doc>',
                'role': 'user',
            },
        ]
        if not stream:
            response = client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content
        else:
            response = client.chat.stream(
                model=model_name,
                messages=messages,
                temperature=temperature
            )
            collected_messages = []
            for res_chunk in response:
                chunk_content = res_chunk.data.choices[0].delta.content
                collected_messages.append(chunk_content)
                print(chunk_content)
            content = ''.join([m for m in collected_messages if m is not None])

        with open(output_path, 'a') as output_file:
            output_file.write(
                f'## {document_name}\n\n{content}\n\n\n\n\n'
            )


def query_with_document(
        document: str,
        prompt: Dict,
        output_path: str,
        stream: Optional[bool] = False,
        client: Optional[Mistral] = None) -> str:

    if not client:
        client = Mistral(
            api_key=os.getenv('MISTRAL_API_KEY', '')
        )

    model_name = prompt.get('model')
    temperature = prompt.get('temperature')

    messages = [
        {
            'content': prompt.get('role'),
            'role': 'system',
        },
        {
            'content': prompt.get('content').replace('[document]', document),
            'role': 'user',
        },
    ]
    if not stream:
        response = client.chat.complete(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        content = response.choices[0].message.content
    else:
        response = client.chat.stream(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        collected_messages = []
        for res_chunk in response:
            chunk_content = res_chunk.data.choices[0].delta.content
            collected_messages.append(chunk_content)
            print(chunk_content)
        content = ''.join([m for m in collected_messages if m is not None])

    with open(output_path, 'w') as output_file:
        output_file.write(content)

    return content


def double_query(
        document: str,
        prompt: Dict,
        output_path: str,
        stream: Optional[bool] = False) -> None:

    client = Mistral(
        api_key=os.getenv('MISTRAL_API_KEY', '')
    )
    model_name = prompt.get('model')
    temperature = 0.2

    base_content = query_with_document(
        document=document,
        prompt=prompt,
        stream=True,
        output_path=output_path
    )

    messages = [
        {
            'content': prompt.get('role'),
            'role': 'system',
        },
        {
            'content': (
                'Rewrite the following letter, '
                'given between <letter></letter> tags, '
                'modifying all the keywords to other '
                'words with equivalent meaning: '
                f'<letter>{base_content}</letter>'
            ),
            'role': 'user',
        },
    ]
    if not stream:
        response = client.chat.complete(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        content = response.choices[0].message.content
    else:
        response = client.chat.stream(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        collected_messages = []
        for res_chunk in response:
            chunk_content = res_chunk.data.choices[0].delta.content
            collected_messages.append(chunk_content)
            print(chunk_content)
        content = ''.join([m for m in collected_messages if m is not None])

    with open(output_path, 'a') as output_file:
        output_file.write('\n\n' + '*' * 100 + '\n\n')
        output_file.write(content)

    return content


if __name__ == '__main__':

    load_dotenv(find_dotenv())

    task = 'redaction'
    version = 'latest'

    prompt_dict, version = read_prompt_file(
        os.getenv('PROMPT_FILE_PATH'),
        task=task,
        version=version
    )
    query_with_document(
        document='',
        prompt=prompt_dict,
        stream=True,
        output_path=os.path.join(
            os.getenv('OUTPUT_DIR_PATH'),
            f'{version}.txt'
        )
    )
