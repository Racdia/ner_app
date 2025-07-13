import streamlit as st
import spacy
from spacy import displacy
import stanza
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import html
import matplotlib.pyplot as plt
import spacy.cli
spacy.cli.download("fr_core_news_md")

@st.cache_resource
def load_spacy_model():
    return spacy.load("fr_core_news_md")

@st.cache_resource
def load_stanza_model():
    stanza.download("fr")
    return stanza.Pipeline(lang="fr", processors="tokenize,ner")

@st.cache_resource
def load_camembert_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner", use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

@st.cache_resource
def load_custom_finetuned_pipeline():
    model_name = "Jean-Baptiste/camembert-ner"  # modèle fine-tuné public
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Fonction pour graphique coloré
def colored_bar_chart(freq_series, label_colors):
    fig, ax = plt.subplots()
    bars = ax.bar(freq_series.index, freq_series.values, color=[label_colors.get(label, "#e0e0e0") for label in freq_series.index])
    ax.set_ylabel("Nombre d'entités")
    ax.set_title("Fréquence par type d'entité")
    return fig

# Configuration de la page
st.set_page_config(page_title="🧠 NER Presse", layout="wide")

# Sidebar
st.sidebar.title("🛠️ Paramètres NER")
model_choice = st.sidebar.selectbox(
    "Choisir le modèle NER",
    ["spaCy", "Stanza", "CamemBERT (HuggingFace)", "Modèle Fine-Tuné"]
)
st.sidebar.markdown("---")

# Titre principal
st.title("📰 Reconnaissance d'Entités Nommées (NER) dans la Presse")
st.markdown(
    "<div style='font-size: 1.05rem'>Déposez votre texte ou article ci-dessous. Les entités (personnes, lieux, organisations...) seront automatiquement détectées, surlignées et présentées dans des onglets.</div>",
    unsafe_allow_html=True
)

text_input = st.text_area("✍️ Entrez un article de presse en français :", height=200)

if st.button("🔍 Analyser"):
    if text_input.strip() != "":
        ents = []
        label_colors = {
            "PER": "#a5d6a7",
            "ORG": "#90caf9",
            "LOC": "#fff59d",
            "MISC": "#ce93d8"
        }

        colored_text = text_input

        if model_choice == "spaCy":
            nlp = load_spacy_model()
            doc = nlp(text_input)
            ents = [
                {"Texte": ent.text, "Type": ent.label_, "Début": ent.start_char, "Fin": ent.end_char}
                for ent in doc.ents
            ]
            offset = 0
            for ent in sorted(doc.ents, key=lambda e: e.start_char):
                ent_text = html.escape(ent.text)
                color = label_colors.get(ent.label_, "#e0e0e0")
                span = f"<span style='background-color: {color}; border-radius: 4px; padding: 0 2px; font-weight: bold'>{ent_text}</span>"
                start = ent.start_char + offset
                end = ent.end_char + offset
                colored_text = colored_text[:start] + span + colored_text[end:]
                offset += len(span) - (end - start)

        elif model_choice == "Stanza":
            nlp = load_stanza_model()
            doc = nlp(text_input)
            ents = [
                {"Texte": ent.text, "Type": ent.type, "Début": ent.start_char, "Fin": ent.end_char}
                for ent in doc.ents
            ]
            offset = 0
            for ent in sorted(doc.ents, key=lambda e: e.start_char):
                ent_text = html.escape(ent.text)
                color = label_colors.get(ent.type, "#e0e0e0")
                span = f"<span style='background-color: {color}; border-radius: 4px; padding: 0 2px; font-weight: bold'>{ent_text}</span>"
                start = ent.start_char + offset
                end = ent.end_char + offset
                colored_text = colored_text[:start] + span + colored_text[end:]
                offset += len(span) - (end - start)

        elif model_choice == "CamemBERT (HuggingFace)":
            pipe = load_camembert_pipeline()
            results = pipe(text_input)
            results = [ent for ent in results if ent.get("start") is not None and ent.get("end") is not None]
            offset = 0
            for ent in sorted(results, key=lambda x: x['start']):
                ent_text = html.escape(ent['word'])
                color = label_colors.get(ent["entity_group"], "#e0e0e0")
                span = f"<span style='background-color: {color}; border-radius: 4px; padding: 0 2px; font-weight: bold'>{ent_text}</span>"
                start = ent['start'] + offset
                end = ent['end'] + offset
                colored_text = colored_text[:start] + span + colored_text[end:]
                offset += len(span) - (end - start)
                ents.append({
                    "Texte": ent["word"],
                    "Type": ent["entity_group"],
                    "Début": ent["start"],
                    "Fin": ent["end"]
                })

        elif model_choice == "Modèle Fine-Tuné":
            pipe = load_custom_finetuned_pipeline()
            results = pipe(text_input)
            results = [ent for ent in results if ent.get("start") is not None and ent.get("end") is not None]
            offset = 0
            for ent in sorted(results, key=lambda x: x['start']):
                ent_text = html.escape(ent['word'])
                color = label_colors.get(ent["entity_group"], "#e0e0e0")
                span = f"<span style='background-color: {color}; border-radius: 4px; padding: 0 2px; font-weight: bold'>{ent_text}</span>"
                start = ent['start'] + offset
                end = ent['end'] + offset
                colored_text = colored_text[:start] + span + colored_text[end:]
                offset += len(span) - (end - start)
                ents.append({
                    "Texte": ent["word"],
                    "Type": ent["entity_group"],
                    "Début": ent["start"],
                    "Fin": ent["end"]
                })

        tab1, tab2, tab3, tab4 = st.tabs(["📄 Texte annoté", "📋 Entités", "📊 Fréquence", "ℹ️ Statistiques modèle"])

        with tab1:
            st.markdown(f"<div style='line-height: 1.6; font-size: 1.1em'>{colored_text}</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 🎨 Légende des couleurs")
            for label, color in label_colors.items():
                st.markdown(f"<span style='background-color: {color}; padding: 4px 8px; border-radius: 4px; margin-right: 6px'>{label}</span>", unsafe_allow_html=True)

        with tab2:
            if ents:
                st.dataframe(pd.DataFrame(ents))
            else:
                st.info("Aucune entité nommée détectée.")

        with tab3:
            if ents:
                freq = pd.Series([e["Type"] for e in ents]).value_counts()
                fig = colored_bar_chart(freq, label_colors)
                st.pyplot(fig)
            else:
                st.info("Pas de données pour le graphique.")

        with tab4:
            if model_choice == "spaCy":
                st.markdown("**Modèle :** `fr_core_news_md` (spaCy)")
                st.markdown("**F1-score NER :** ~85% (données WikiNER)")
            elif model_choice == "Stanza":
                st.markdown("**Modèle :** `stanza fr` (UD French-GSD)")
                st.markdown("**F1-score NER :** ~80% selon les benchmarks UD")
            elif model_choice == "CamemBERT (HuggingFace)":
                st.markdown("**Modèle :** `wikineural-multilingual-ner`")
                st.markdown("**Langues :** multilingue, fine-tuné sur WikiANN")
                st.markdown("**F1-score moyen :** ~86% (langues européennes)")
            elif model_choice == "Modèle Fine-Tuné":
                st.markdown("**Modèle :** `Jean-Baptiste/camembert-ner`")
                st.markdown("**Corpus :** WikiNER français")
                st.markdown("**F1-score :** ~87.6% sur données WikiNER")
    else:
        st.warning("Veuillez saisir un texte.")
