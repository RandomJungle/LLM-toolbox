import json
import os
from typing import List, Optional, Dict, Tuple

import pandas as pd
import pdfplumber


def read_json_dataframe(
        file_path: str,
        remove_duplicates: bool,
        orient: Optional[str] = 'table') -> pd.DataFrame:
    if file_path.endswith('.jsonl'):
        dataframe = pd.read_json(
            path_or_buf=file_path,
            lines=True
        )
    else:
        dataframe = pd.read_json(
            path_or_buf=file_path,
            orient=orient
        )
    if remove_duplicates:
        dataframe = dataframe.drop_duplicates(subset='id')
    return dataframe


def read_prompt_file(
        file_path: str,
        task: str,
        version: Optional[str] = 'latest') -> Tuple[Dict, str]:
    with open(file_path, 'r') as prompt_file:
        prompt_dict = json.loads(prompt_file.read())
    task_dict = prompt_dict.get(task)
    if version == 'latest':
        version = str(max(
            [float(key) for key in task_dict.keys() if not 'template' in key]
        ))
        selected_prompt = task_dict.get(version)
    else:
        selected_prompt = task_dict.get(version)
    return selected_prompt, version


def convert_pdf2txt(pdf_path: str):
    texts = []
    with pdfplumber.open(pdf_path) as pdf_file:
        for page in pdf_file.pages:
            texts.append(page.extract_text())
    return ('\n' * 10).join(texts)


def read_pdfs(bibliography_path: str):
    bibliography = dict()
    for file_name in os.listdir(bibliography_path):
        bibliography[file_name] = convert_pdf2txt(
            pdf_path=os.path.join(bibliography_path, file_name)
        )
    return bibliography


def chunk_dataframe(
        dataframe: pd.DataFrame,
        num_chunks: int) -> List[pd.DataFrame]:
    """
    Split dataframe into n chunks (mainly for LLM querying)
    """
    if num_chunks <= 1:
        chunks = [dataframe]
    else:
        rows_per_chunk = len(dataframe) // num_chunks
        chunks = [
            dataframe[i * rows_per_chunk:(i + 1) * rows_per_chunk]
            for i in range(num_chunks)
        ]
        if len(dataframe) % num_chunks != 0:
            chunks.append(dataframe[num_chunks * rows_per_chunk:])
    return chunks