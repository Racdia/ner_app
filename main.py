# This is a sample Python script.
from transformers import CamembertTokenizer, AutoModelForTokenClassification, pipeline


# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    text = """Emmanuel Macron s’est rendu à Marseille ce lundi 12 juin 2023 afin d’assister à une conférence organisée par l’Organisation Mondiale de la Santé. Lors de sa visite, le président de la République a rencontré le maire Benoît Payan et a annoncé un investissement de 100 millions d’euros pour la rénovation des hôpitaux de la ville. Plusieurs représentants de la société Sanofi étaient également présents lors de la conférence qui s’est tenue au Palais du Pharo."""

    tokenizer = CamembertTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    results = ner(text)
    for r in results:
        print(r)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
