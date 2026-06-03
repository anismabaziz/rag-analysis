from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader


def load_documents(path: str):
    file_extractor = {".pdf": PyMuPDFReader()}
    return SimpleDirectoryReader(input_dir=path, file_extractor=file_extractor).load_data() 