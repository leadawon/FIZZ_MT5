import json
import torch
import spacy
import nltk
from tqdm import tqdm
from nltk import sent_tokenize
from collections import defaultdict
from transformers import MT5Tokenizer, T5ForConditionalGeneration

from state import State
from util import create_document, create_next_batch, extract_result_string, predict_coreferences

# 1. 모델 및 전처리기 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer_nltk = nltk.WordPunctTokenizer()
nlp = spacy.load("en_core_web_lg")

model_ckpt = "mt5-coref-pytorch/link-append-xxl"
tokenizer = MT5Tokenizer.from_pretrained(model_ckpt, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_ckpt, torch_dtype=torch.float16).to(device)

# 2. 대명사 목록
pronouns = ["i", "he", "she", "you", "me", "him", "myself", "yourself", "himself", "herself", "yourselves"]
special_pronouns = ["my", "mine", "her", "hers", "his", "your", "yours"]

# 3. Coreference Resolution 함수
def resolve_coreferences(doc: str) -> str:
    inputs = [{'document_id': 'doc', 'sentences': []}]
    for sentence in sent_tokenize(doc):
        inputs[0]['sentences'].append({'speaker': '_', 'words': tokenizer_nltk.tokenize(sentence)})

    states_dict = {}
    for doc in inputs:
        states_dict[doc['document_id']] = State(create_document(doc), tokenizer)

    while True:
        states, batches = create_next_batch(states_dict)
        if not states:
            break
        predictions = predict_coreferences(tokenizer, model, batches, len(batches))
        results = extract_result_string(predictions)
        for state, result, batch in zip(states, results, batches):
            state.extend(result)

    for doc_name, s in states_dict.items():
        resolved_text = []
        all_clusters = [cluster for _, cluster in s.cluster_name_to_cluster.items()]
        text, text_map = [], []
        for k, snt in s.input_document['sentences'].items():
            m = s.input_document['token_maps'][k]
            text += snt
            text_map += m

        words_dict = {}
        pred_clusters = []
        for cluster in all_clusters:
            has_person = False
            for st, en in cluster:
                mention = " ".join(text[st:en+1]).title()
                mention_nlp = nlp(mention)
                for ent in mention_nlp.ents:
                    if ent.label_ == "PERSON":
                        cluster_id = s.mention_index_to_cluster_name[str((st, en))]
                        words_dict[cluster_id] = ent.text.replace("'s", "").strip()
                        has_person = True
                        break
            if has_person:
                pred_clusters.append(cluster)

        cluster_annotations_start = [[] for _ in text_map]
        cluster_annotations_end = [[] for _ in text_map]
        for ci in pred_clusters:
            for m in ci:
                for i, tid in enumerate(text_map):
                    if tid == m[0]:
                        m_len = m[1] - m[0]
                        name = s.mention_index_to_cluster_name[str(m)]
                        cluster_annotations_start[i].append((name, m_len))
                    if tid == m[1]:
                        cluster_annotations_end[i].append(']')

        resolved_text = []
        for tok, start in zip(text, cluster_annotations_start):
            is_resolved = False
            if start:
                x = start[0]
                lower_tok = tok.lower()
                try:
                    if lower_tok in pronouns:
                        resolved_text.append(words_dict[x[0]])
                        is_resolved = True
                    elif lower_tok in special_pronouns:
                        resolved_text.append(words_dict[x[0]] + "'s")
                        is_resolved = True
                    else:
                        tok_nlp = nlp(tok)
                        for ent in tok_nlp.ents:
                            if ent.label_ in ["PERSON", "ORG"]:
                                break
                        else:
                            resolved_text.append(words_dict[x[0]] + ',')
                            is_resolved = True
                except:
                    pass
            if not is_resolved:
                resolved_text.append(tok.lower())

    return ' '.join(resolved_text)

# 4. JSON 처리 함수
def process_and_save_coref(input_path: str, output_path: str):
    print(f"Processing: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    resolved_data = []
    for entry in tqdm(data):
        resolved_entry = {
            "document": resolve_coreferences(entry["document"]),
            "claim": resolve_coreferences(entry["claim"]),
            "label": entry["label"],
            "cut": entry["cut"]
        }
        resolved_data.append(resolved_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resolved_data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}\n")

# 5. 실행
input_files = [
    "aggre_fact_cnndm_sota.json",
    "aggre_fact_xsum_sota.json",
    "aggregated_cnndm_final.json",
    "aggregated_xsum_final.json"
]

for input_file in input_files:
    output_file = input_file.replace(".json", "_coref.json")
    process_and_save_coref(input_file, output_file)
