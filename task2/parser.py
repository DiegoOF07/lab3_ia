import re

def read_dataset(path: str) -> list[str]:
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            label, message = line.strip().split(maxsplit=1)
            data.append((label, message))
    return data

def clean_message(message: str) -> list[str]:
    message = message.lower()
    message = re.sub(r"[^a-záéíóúñü\s]", "", message)
    return message.split()

def build_vocabulary(data):
    vocabulary = set()
    for _, message in data:
        vocabulary.update(clean_message(message))
    return vocabulary
