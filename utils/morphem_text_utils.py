def strip_pos_tags(text):
    tokens = []
    for token in text.split():
        sep_index = token.rfind("/")
        assert sep_index != -1, f"input text must be output of morpheme analyzer"
        tokens.append(token[:sep_index])

    return " ".join(tokens)


def strip_bi_tags(text):
    tokens = []
    for token in text.split():
        sep_index = token.rfind("/")

        assert token[sep_index: sep_index + 3] in ("/B-", "/I-")

        morp = token[:sep_index]
        pos_tag = token[sep_index + 3:]

        if morp:
            tokens.append(f"{morp}/{pos_tag}")

    return " ".join(tokens)


if __name__ == "__main__":
    exit()

