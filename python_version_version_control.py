# %% [markdown]
# # 2025 COMP90042 Project
# *Make sure you change the file name with your group id.*

# %% [markdown]
# # Readme (EDITED BY JOHN and Guwei)
# 
# *If there is something to be noted for the marker, please mention here.*
# 
# *If you are planning to implement a program with Object Oriented Programming style, please put those the bottom of this ipynb file*

# %% [markdown]
# # 1.DataSet Processing
# (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)

# %%
%pip install sentence-transformers
%pip install faiss-cpu
%pip install faiss-gpu
%pip install datasets
%pip install gdown
%pip install accelerate
%pip install transformers[torch]
%pip install huggingface_hub[hf_xet]
%pip install seaborn
%pip install matplotlib

# %%
import gdown
import os

# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# # File Path
# data_dir = "/content/drive/MyDrive/COMP90042-Data"


# URLs of the files to download
urls = [
    "https://drive.google.com/uc?id=1SIlHpjPhgr5NJpf6nK79aCevoOmvZgML",
    "https://drive.google.com/uc?id=1aTH-Zzq9dztxxXIPDW8MY3HZR_6pGbSt",
    "https://drive.google.com/uc?id=1iliVuUhHp2M48Svxl2FUa4hWaOxnnTo7",
    "https://drive.google.com/uc?id=1-AWc4xhd7YVV45tOti9Uu778x4YCsyeZ",
    "https://drive.google.com/uc?id=1zvWH6i6EQwTVgFlaKfNm6TXeo6DEEA9m"
]

# Corresponding filenames
filenames = [
    "evidence.json",
    "dev-claims.json",
    "train-claims.json",
    "test-claims-unlabelled.json",
    "dev-claims-baseline.json"
]

# Download each file
for url, filename in zip(urls, filenames):
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists. Skipping download.")



# %% [markdown]
# Imports

# %%
import faiss
from sentence_transformers import CrossEncoder
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
import random
import torch
import pandas as pd
import json
import re
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from tqdm import tqdm

# %%
def retrieve(evi_ebds, claim_ebds, evi_df, claim_df, retrival_top_k, rerank_top_k,threshold_activated,score_threshold, cross_encoder,dev):
    embedding_dim = evi_ebds.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    faiss.normalize_L2(evi_ebds)
    index.add(evi_ebds)
    retrieval = pd.DataFrame()
    retrieved_labels = []
    retrieved_evidences = []

    i = 0
    counts = 0
    claim_texts = []
    total = len(claim_ebds)
    for dev_claim_embedding in claim_ebds:
        faiss.normalize_L2(dev_claim_embedding.reshape(1, -1))
        D, I = index.search(dev_claim_embedding.reshape(1, -1), retrival_top_k)
        text = claim_df.iloc[i]['claim_text']
        pairs = [(text, evi_df['value'][a]) for a in I[0]]
        scores = cross_encoder.predict(pairs)
        reranked = sorted(zip(I[0], scores), key=lambda x: x[1], reverse=True)

        retrieved_evidence = []
        filtered = []
        first = True
        for idx, score in reranked[:rerank_top_k]:
            if first:
                evi_id = evi_df['ID'][idx]
                evi_content = evi_df['value'][idx]
                retrieved_evidence.append(evi_id)
                filtered.append((evi_id, score, evi_content))
                first = False
            else:
                if not threshold_activated or score >= score_threshold:
                    evi_id = evi_df['ID'][idx]
                    evi_content = evi_df['value'][idx]
                    retrieved_evidence.append(evi_id)
                    filtered.append((evi_id, score, evi_content))

        retrieved_evi =  [evi_df['ID'][a] for a in I[0]]
        retrieved_evidences.append(retrieved_evidence)
        retrieved_labels.append('SUPPORTS')
        claim_texts.append(text)

        if dev:
            print(f"Claim: {text}")
            if len(filtered) > 0:
                print("evidence relevant")
                for eid, score, c in filtered:
                    print(f"  {eid}, Score: {score:.4f} ")
            print(f"Ground truth: {claim_df.iloc[i]['evidences']}\n")

            count = 0
            for g in claim_df.iloc[i]['evidences']:
                if g in retrieved_evi:
                    count += 1
            counts += count / len(claim_df.iloc[i]['evidences'])

        print('Progress ', round(i * 100 / total, 3), '%')
        i += 1

    print("R: ", counts/ i)
    
    # Create the dataframe with all necessary information
    retrieval['ID'] = claim_df['ID'].values[:len(retrieved_evidences)]
    retrieval['evidences'] = retrieved_evidences
    retrieval['claim_label'] = retrieved_labels
    retrieval['claim_text'] = claim_texts
    
    return retrieval

def mine_hard_negatives(retrieved_evidences, ground_truth_evidences):
    negative_evidences = []
    for i in range(len(retrieved_evidences)):
        retrieved_evidence = retrieved_evidences.iloc[i]
        ground_truth_evidence = ground_truth_evidences.iloc[i]
        negative_evidence = []
        for e in retrieved_evidence:
            if e not in ground_truth_evidence:
                negative_evidence.append(int(re.findall(r'\d+', e)[0]))
        negative_evidences.append(negative_evidence)
    return negative_evidences

def load_train_data(train_dataframe, evidence_dateframe, random_negatives_amount_per_claim):
    train_data = []
    for id in range(len(train_dataframe)):
        claim_text =  train_dataframe.iloc[id]['claim_text']
        # print("CLAIM: ", claim_text)
        positive_evidence_ids = train_dataframe.iloc[id]['evidences_numeric_index']
        negative_evidence_ids = train_dataframe.iloc[id]['negative_evidences']
        for evid_id in positive_evidence_ids:

            evidence_text = evidence_dateframe.iloc[evid_id]['value']
            # print("POSITIVE: ", evidence_text)
            train_data.append(InputExample(texts=[claim_text, evidence_text], label=1.0))
        for ngevid_id in negative_evidence_ids:
            evidence_text = evidence_dateframe.iloc[ngevid_id]['value']
            # print("NEGATIVE: ", evidence_text)
            train_data.append(InputExample(texts=[claim_text, evidence_text], label=0.0))

        for i in range(random_negatives_amount_per_claim):
            neg_id = random.choice(list(set(evidence_dateframe.index) - set(positive_evidence_ids)))
            neg_text = evidence_dateframe.iloc[neg_id]['value']
            train_data.append(InputExample(texts=[claim_text, neg_text], label=0.0))
    return train_data

def load_positive_train_data(train_dataframe, evidence_dateframe):
    train_data = []
    for id in range(len(train_dataframe)):
        claim_text =  train_dataframe.iloc[id]['claim_text']
        positive_evidence_ids = train_dataframe.iloc[id]['evidences_numeric_index']
        for evid_id in positive_evidence_ids:
            evidence_text = evidence_dateframe.iloc[evid_id]['value']
            train_data.append(InputExample(texts=[claim_text, evidence_text], label=1.0))
    return train_data

def save_retrieval(retrieval, path):
    output = {}
    for i in range(len(retrieval)):
        output[retrieval.iloc[i]['ID']] = {}
        output[retrieval.iloc[i]['ID']]['evidences'] = retrieval.iloc[i]['evidences']
        output[retrieval.iloc[i]['ID']]['claim_label'] = retrieval.iloc[i]['claim_label']
        output[retrieval.iloc[i]['ID']]['claim_text'] = retrieval.iloc[i]['claim_text']

    with open(path, 'w') as file:
        file.write(json.dumps(output))

# %% [markdown]
# Read evidence dataset.

# %%
with open('evidence.json', 'r') as f:
    evidence = json.load(f)
flat_list = []
for key in evidence:
    flat_list.append({"ID": key, "value": evidence[key]})

evidence_df = pd.DataFrame(flat_list)

evidence_df.head()

# %% [markdown]
# Read training and developing datasets

# %%
with open('train-claims.json', 'r') as f:
    train_df = json.load(f)
flat_list = []
for key in train_df:
    evidences_numeric_index = []
    for e in train_df[key]['evidences']:
        evidences_numeric_index.append(int(re.findall(r'\d+', e)[0]))
    flat_list.append({"ID": key, "claim_text": train_df[key]['claim_text'], "claim_label": train_df[key]['claim_label'], "evidences": train_df[key]['evidences'], "evidences_numeric_index": evidences_numeric_index})
train_df = pd.DataFrame(flat_list)

train_df.head()



# %%
with open('dev-claims.json', 'r') as f:
    dev_df = json.load(f)
flat_list = []
for key in dev_df:
    evidences_numeric_index = []
    for e in dev_df[key]['evidences']:
        evidences_numeric_index.append(int(re.findall(r'\d+', e)[0]))
    flat_list.append({"ID": key, "claim_text": dev_df[key]['claim_text'], "claim_label": dev_df[key]['claim_label'], "evidences": dev_df[key]['evidences'], "evidences_numeric_index": evidences_numeric_index})
dev_df = pd.DataFrame(flat_list)

dev_df.head()

# %% [markdown]
# # 2. Model Implementation
# (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)

# %% [markdown]
# Calculate sentence embeddings for evidence dataset and trainning dataset.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

train_data = load_positive_train_data(train_df, evidence_df)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    output_path='./output-bi-encoder',
    weight_decay=0.01
)

evidence_embeddings = model.encode(evidence_df['value'], batch_size=64, show_progress_bar=True)
dev_claims_embeddings = model.encode(dev_df['claim_text'], batch_size=64, show_progress_bar=True)
train_claims_embeddings = model.encode(train_df['claim_text'], batch_size=64, show_progress_bar=True)


# %% [markdown]
# Fine tune cross encoder for re-ranking.

# %%
from datasets import Dataset
def load_positive_train_ds(train_dataframe, evidence_dateframe, t_ebds, e_ebds):
    query = []
    answer = []
    for id in range(len(train_dataframe)):
        claim_text =  train_dataframe.iloc[id]['claim_text']
        positive_evidence_ids = train_dataframe.iloc[id]['evidences_numeric_index']
        # claim_ebd = t_ebds[id]
        # none_added = True
        for evid_id in positive_evidence_ids:

            # evi_ebd = e_ebds[evid_id]
            evidence = evidence_dateframe.iloc[evid_id]['value']
            # similarities = model.similarity(claim_ebd, evi_ebd)

            # if similarities > 0:
            #     none_added = False
            # print('Claim: ', claim_text)
            # print('Evidence: ', evidence)

            query.append(claim_text)
            answer.append(evidence)
        # if none_added:
        #     query.append(claim_text)
        #     answer.append(evidence_dateframe.iloc[positive_evidence_ids[0]]['value'])

    return Dataset.from_dict({
    "query": query,
    "answer": answer,})


train_dataset = load_positive_train_ds(train_df, evidence_df, train_claims_embeddings, evidence_embeddings)
print(len(train_dataset))

# %%
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments, CrossEncoderTrainer, losses
from sentence_transformers.util import mine_hard_negatives

eval_dataset = load_positive_train_ds(dev_df, evidence_df,dev_claims_embeddings, evidence_embeddings)
embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
hard_eval_dataset = mine_hard_negatives(
    eval_dataset,
    embedding_model,
    corpus = train_dataset["answer"],  # Use the full dataset as the corpus
    num_negatives=50,  # How many negatives per question-answer pair
    batch_size=4096,  # Use a batch size of 4096 for the embedding model
    output_format="n-tuple",  # The output format is (query, positive, negative1, negative2, ...) for the evaluator
    include_positives=True,  # Key: Include the positive answer in the list of negatives
    use_faiss=True,  # Using FAISS is recommended to keep memory usage low (pip install faiss-gpu or pip install faiss-cpu)
)

reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=[
        {
            "query": sample["query"],
            "positive": [sample["answer"]],
            "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
        }
        for sample in hard_eval_dataset
    ],
    batch_size=32,
    name="gooaq-dev",
)

# %%
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

reranking_evaluator(cross_encoder)


model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"reranker-{short_model_name}-gooaq-cmnrl"
args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs= 40,
    per_device_train_batch_size= 16,
    per_device_eval_batch_size= 16,
    learning_rate=1e-6,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="epoch",
    save_strategy="epoch",

    save_total_limit=2,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    seed=12,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    weight_decay = 0.01,
)
loss = losses.MultipleNegativesRankingLoss(model=cross_encoder)

trainer = CrossEncoderTrainer(
    model=cross_encoder,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=reranking_evaluator,
    eval_dataset=eval_dataset,
)
trainer.train()

# cross_encoder.fit(

#     train_dataloader=train_dataloader,
#     # loss_fct=train_loss,
#     epochs=20,
#     output_path='./fine-tuned-cross-encoder',
#     weight_decay=0.01
# )

# %% [markdown]
# Second retrieve for testing.

# %%
# train_retrieval = retrieve(evidence_embeddings, train_claims_embeddings, evidence_df , train_df, 150, 4, True, 2, cross_encoder, True)
# save_retrieval(train_retrieval, 'train_retrieval.json')

# %%

# dev_retrieval = retrieve(evidence_embeddings, dev_claims_embeddings, evidence_df , dev_df, 150, 5, True, 0, cross_encoder, True)
# save_retrieval(dev_retrieval, 'dev_retrieval.json')


# %% [markdown]
# # 3.Testing and Evaluation
# (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)

# %% [markdown]
# I will do a classficiation, we were thinking of using transformer model then LSTM, and compare them and use the one with the best performance. 
# For transformer, in this fact checking task, I will first try with the multi encoder fusion because we have a lot of cluses in the evidence and want to combine 
# 

# %%
with open('test-claims-unlabelled.json','r') as f:
    data = f.read()
print(data[:200])

test_df = json.loads(data)
flat_list = []
for key in test_df:
    flat_list.append({"ID": key, "claim_text": test_df[key]['claim_text']})
tst_df = pd.DataFrame(flat_list)
tst_claims_embeddings = model.encode(tst_df['claim_text'], batch_size=64, show_progress_bar=True)
retrieval = retrieve(evidence_embeddings, tst_claims_embeddings, evidence_df , tst_df, 150, 5, False, 0, cross_encoder, False)
save_retrieval(retrieval, 'test-output.json')


# %%
with open('test-output.json', 'r') as f:
    test_output = json.load(f)


# %% [markdown]
# I am going to initialise a transformer model and tokensizer for the evidence classification task.
# My plan is to use fastest training times and speed as this is done from gooogle collab and our retrival already takes a lot of time which almost is 6 hours to train in colab unfortunately. We have enough disk space but I want to keep the training simple and most generic as possible. 
# There were few options, but we will go with distilbert-base-uncased, which is a smaller version of BERT. 
# 
# Decisions based on cased and uncased is uncased because itt is faster and no need to worry about capital named entities.
# {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED} There are 4 classes in the evidence classification task.
# 
# I also want to keep the label order and everything simple as possible.
# 

# %%
# import that wasn't done in the beginning
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model Choice and Label List Initialization
classifer_model = "distilbert-base-uncased"
label_list=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
num_labels = len(label_list)
label_map = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print("Label Map: ", label_map)
print("ID to Label Map: ", id2label)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(classifer_model)
model = AutoModelForSequenceClassification.from_pretrained(classifer_model, num_labels=num_labels, id2label=id2label, label2id=label_map)

# Load the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# For preparing training data from claims for the ground truth evidence, I will be using  one training instance per evidence.
# E.g.
# 1. (claim_tex [SEP] evidence_text_1, "SUPPORTS")
# 2. (claim_text [SEP] evidence_text_2, "SUPPORTS")
# 3. (claim_text [SEP] evidence_text_3, "REFUTES")
# 4. (claim_text [SEP] "NOT_ENOUGH_INFO")
# 5. (claim_text [SEP] evidence_text_5, "DISPUTED")
# 
# Kind of ways, because this is the best way to train data for the model.
# 
# Some options we considered were: only using one of the randomly selected evidence or first evidence.
# Or put all evidenece in one instance.
# 
# We will also use PyTorch for the training to do it manually instead of huggingface trainer.
# 
# Hyperparameters for the training:
# - learning_rate = ADAMW(1e-5) Optimizer
# - batch_size = 16
# - epochs = 10
# 
# We will use dev-claims for validation and test-claims for testing.
# 
# We will try to look for f score and accuracy for the evaluation.
# 
# For the ones that has label but no listed evidence, we will create a special input format like
# (claim_text [SEP], "NOT_ENOUGH_INFO") since we are not allowed to modify the dataset or can't skip it written in the project description.This may be a problem with imbalance dataset, however, at the end I am relying on retrival, and if the retrival is good with no evidence, then I will have to make it not enough info
# 
# claims labllee
# 
# Also later changed to tokenizers real separator (tokenizer.sep_token) in order to learn the model better.
# 
# Also I added a checkpoint to train fixed 10 epochs and save the best model by tracking the best accuracy.

# %%
# custom pytorch dataset class
class ClaimEvidenceDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.texts = [item['text'] for item in data_list]
        self.labels = [item['label_id'] for item in data_list]
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# function to match the evidence with the claim and prepare the dataset
def prepare_data(df, evidence_df, label_map):
    items = []
    for _, row in df.iterrows():
        claim = row['claim_text']
        for evid_id in row.get('evidences', []):
            evid_row = evidence_df[evidence_df['ID'] == evid_id]
            if evid_row.empty: 
                # show if evidence is not found
                print(f"Evidence ID {evid_id} not found for claim ID {row['ID']}")
                continue

            else:
                text = f"{claim} {tokenizer.sep_token} {evid_row['value'].iloc[0]}"
                label = label_map[row['claim_label']]
                items.append({'text': text, 'label_id': label})
    return items

# build datasets and dataloaders
train_items = prepare_data(train_df, evidence_df, label_map)
dev_items   = prepare_data(dev_df,   evidence_df, label_map)

# print to debug it is working simple 
print("Done preparing daata: ", len(train_items), " train items and ", len(dev_items), " dev items")

train_dataset = ClaimEvidenceDataset(train_items, tokenizer)
dev_dataset   = ClaimEvidenceDataset(dev_items,   tokenizer)

# # print to debug the claimevidence dataset is working
# print("Train Dataset: ", train_dataset[0])
# print("Dev Dataset: ", dev_dataset[0])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader   = DataLoader(dev_dataset,   batch_size=16)



# %%
# training loop (3 epochs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# variables for training
num_epochs = 10
best_f1 = 0
best_model_state = None

# optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # add progress bar for batches
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # update progress bar with current loss
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train loss: {avg_train_loss:.4f}")

    # evaluation on dev set
    model.eval()
    preds, trues = [], []
    eval_progress = tqdm(dev_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Evaluating)")
    with torch.no_grad():
        for batch in eval_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            trues.extend(labels.cpu().tolist())

    acc = accuracy_score(trues, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average='weighted')
    print(f"Epoch {epoch+1} Dev acc: {acc:.4f}, F1: {f1:.4f}")
    print(classification_report(trues, preds, target_names=label_map.keys()))

    # Save the best model based on F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict().copy()
        print(f"New best model saved with F1: {f1:.4f}")

# Load the best model state
if best_model_state:
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "claim_evidence_classifier_best.pth")
    print(f"Best model saved with F1: {best_f1:.4f}")

# %% [markdown]
# Now I am going to load the best model state to store in state_dict to a file
# 
# Also I will now need to prepare and batch the test data for prediction and for that I will use pytorch dataset and dataloader for convenience consistency and speed as google collab is slow

# %%


# %%
# evidence_df
# dev_df
# train_df
# tst_df
# test_df

# %%
# # Save the model for later use
# torch.save(model.state_dict(), "claim_evidence_classifier.pth")
# print("Model saved successfully!")

# A function to prepare test data
def prepare_test_data(claim_text, evidences, evidence_df, tokenizer):
    items = []
    for evid_id in evidences:
        evid_row = evidence_df[evidence_df['ID'] == evid_id]
        if evid_row.empty:
            continue
        text = f"{claim_text} {tokenizer.sep_token} {evid_row['value'].iloc[0]}"
        items.append({'text': text})
    return items

# Function to predict labels
def predict_claim_label(model, items, tokenizer, device):
    model.eval()
    
    # If no evidence, return NOT_ENOUGH_INFO
    if not items:
        return "NOT_ENOUGH_INFO"
    
    # Create a dataset and dataloader for the items
    class PredictionDataset(Dataset):
        def __init__(self, items, tokenizer, max_len=128):
            self.tokenizer = tokenizer
            self.texts = [item['text'] for item in items]
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
    
    dataset = PredictionDataset(items, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8)
    
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(preds)
    
    # Count occurrences of each label
    counts = {}
    for pred in predictions:
        label = id2label[pred]
        counts[label] = counts.get(label, 0) + 1
    
    # Return the most common label, or if tie, prioritize in order: SUPPORTS, REFUTES, DISPUTED, NOT_ENOUGH_INFO
    if not counts:
        return "NOT_ENOUGH_INFO"
    
    max_count = max(counts.values())
    max_labels = [label for label, count in counts.items() if count == max_count]
    
    priority_order = ["SUPPORTS", "REFUTES", "DISPUTED", "NOT_ENOUGH_INFO"]
    for label in priority_order:
        if label in max_labels:
            return label
    
    return max_labels[0]  # Fallback

# Load the test output
with open('test-output.json', 'r') as f:
    test_output = json.load(f)

# Predict labels for each claim
print("Predicting labels for test claims...")
for claim_id, claim_data in tqdm(test_output.items()):
    claim_text = claim_data['claim_text']
    evidences = claim_data['evidences']
    
    # Prepare data for this claim
    items = prepare_test_data(claim_text, evidences, evidence_df)
    
    # Get prediction
    predicted_label = predict_claim_label(model, items, tokenizer, device)
    
    # Update the label
    test_output[claim_id]['claim_label'] = predicted_label

# Save the updated test output
with open('test-output-updated.json', 'w') as f:
    json.dump(test_output, f, indent=2)

print("Updated test-output.json with predicted labels!")

# %% [markdown]
# ## Object Oriented Programming codes here
# 
# *You can use multiple code snippets. Just add more if needed*


