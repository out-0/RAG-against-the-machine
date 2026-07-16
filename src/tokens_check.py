import tiktoken

def tokens_count_from_string(string: str, encoding_name: str) -> int:
    """Calculate the count of tokens that resulted from encoding a string
    
    string: The text to be encoded
    encoding_name: Tokenizer name
    """

    encoding: tiktoken.Encoding = tiktoken.get_encoding(
        encoding_name=encoding_name
    )
    tokens_count: int = len(encoding.encode(string))
    return tokens_count
