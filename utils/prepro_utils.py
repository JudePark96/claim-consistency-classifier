import re


def normalize_question(question: str) -> str:
    if question[-1] == '?' or question[-1] == '.':
        question = question[:-1].strip()
    return question


def is_whitespace(c: str) -> bool:
    return bool(re.match("\s", c))


def convert_text_to_ids_string(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return "".join(["[" + str(i) + "]" for i in token_ids])


def convert_tokens_to_ids_string(tokens, tokenizer):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return "".join(["[" + str(i) + "]" for i in token_ids])


def convert_ids_string_to_token_ids(ids_string):
    ids_string = ids_string.replace("[", " ").replace("]", " ")
    return [*map(int, ids_string.strip().split())]


if __name__ == "__main__":
    exit()
