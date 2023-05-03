import csv

import torch, os, re, pandas as pd, json
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, GPT2Tokenizer, GPT2LMHeadModel, \
    Trainer, TrainingArguments, AutoConfig
from datasets import Dataset
import ujson
import itertools
from tqdm import tqdm
import json

# special tokens are defined
bos = '<|endoftext|>'
eos = '<|EOS|>'
body = '<|body|>'
additional_special_tokens = [body]

special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': '<pad>',
                       'sep_token': body}

base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if torch.cuda.is_available():
    print("CUDA available")
    dev = "cuda"
else:
    print("CUDA not available")
    dev = "cpu"
device = torch.device(dev)
print(device)


def get_base_model():
    # the new token is added to the tokenizer
    num_added_toks = base_tokenizer.add_special_tokens(special_tokens_dict)

    # model configuration to which we add the special tokens
    config = AutoConfig.from_pretrained('gpt2',
                                        bos_token_id=base_tokenizer.bos_token_id,
                                        eos_token_id=base_tokenizer.eos_token_id,
                                        pad_token_id=base_tokenizer.pad_token_id,
                                        sep_token_id=base_tokenizer.sep_token_id,
                                        output_hidden_states=False)

    # we load the pre-trained model with custom settings
    base_model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

    # model embeding resizing
    base_model.resize_token_embeddings(len(base_tokenizer))

    return base_model


def tokenize_function(examples):
    return base_tokenizer(examples['text'], padding=True)


def train(news_df, base_model):
    # We add the tokens
    prepare_text = lambda x: ' '.join([bos, x['title'], body, x['abstract'], eos])
    news_df['text'] = news_df.apply(prepare_text, axis=1)

    # Split in train and test
    df_train_news, df_val_news = train_test_split(news_df, train_size=0.9, random_state=77)

    # we load the datasets from pandas df
    train_dataset = Dataset.from_pandas(df_train_news[['text']])
    val_dataset = Dataset.from_pandas(df_val_news[['text']])

    # tokenization
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1
    )

    tokenized_val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1
    )

    model_articles_path = './news-articles_v1'

    training_args = TrainingArguments(
        output_dir=model_articles_path,  # output directory
        num_train_epochs=5,  # total # of training epochs
        per_device_train_batch_size=5,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=200,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=model_articles_path,  # directory for storing logs
        prediction_loss_only=True,
        save_steps=10000
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=base_model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,  # training dataset
        eval_dataset=tokenized_val_dataset,  # evaluation dataset

    )

    trainer.train()
    trainer.save_model()
    base_tokenizer.save_pretrained(model_articles_path)


def generate_n_text_samples(model, tokenizer, input_text, device, n_samples=5):
    text_ids = tokenizer.encode(input_text, return_tensors='pt')
    text_ids = text_ids.to(device)
    model = model.to(device)

    generated_text_samples = model.generate(
        text_ids,
        max_length=400,
        num_return_sequences=n_samples,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_p=0.92,
        temperature=.85,
        do_sample=True,
        top_k=125,
        early_stopping=True
    )
    gen_text = []
    for t in generated_text_samples:
        text = tokenizer.decode(t, skip_special_tokens=True)
        gen_text.append(text)

        return gen_text


def pretty_print(text, max_len_line=200):
    words = text.split(' ')
    len_line = 0
    line = ''
    for w in words:
        if w == '\n':
            print(line)
            line = ''
            continue
        if (len(line) + len(w)) > max_len_line:
            print(line)
            line = ''
        line += ' ' + w
    print(line)


def predict():
    headlines = ["Artificial Intelligence in Cyber Security "]  # change text for different generations
    # trained model loading
    model_articles_path = './news-articles_v1'

    news_model = GPT2LMHeadModel.from_pretrained(model_articles_path)
    news_tokenizer = GPT2Tokenizer.from_pretrained(model_articles_path)

    bos = news_tokenizer.bos_token
    eos = news_tokenizer.eos_token
    # print(f"eos: {eos} bos: {bos}")
    # print(news_tokenizer.additional_special_tokens)
    # sep = news_tokenizer.additional_special_tokens[0]
    sep = additional_special_tokens[0]

    articles = {}
    for headline_raw in headlines:
        headline = ' '.join([bos, headline_raw, sep])
        abstract = generate_n_text_samples(news_model, news_tokenizer, headline,
                                           device, n_samples=5)[0]
        articles[headline_raw] = abstract.replace(headline_raw, '')

    for title, abstract in articles.items():
        print('\033[1m' + title + '\033[0m')
        pretty_print(abstract)
        print()


# we will consider below 3 categories for training
paper_categories = ["cs.AI",  # Artificial Intelligence
                    "cs.CV",  # Computer Vision and Pattern Recognition
                    "cs.LG"]  # Machine Learning


data_file = 'data/arxiv-metadata-oai-snapshot.json'

""" Using `yield` to load the JSON file in a loop to prevent Python memory issues if JSON is loaded directly"""


def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line


def build_dataset(categories=None):
    """
    Convert JSON file to Pandas dataframe (with Kaggle dataset contributor`s code)
    :param categories: desired categories for text generation
    :return:
    """
    if categories is None:
        categories = paper_categories
    titles = []
    abstracts = []
    metadata = get_metadata()
    for paper in tqdm(metadata):
        paper_dict = json.loads(paper)
        category = paper_dict.get('categories')
        if "cs" in category:
            try:
                year = int(paper_dict.get('journal-ref')[-4:])
                titles.append(paper_dict.get('title'))
                abstracts.append(paper_dict.get('abstract').replace("\n", ""))
            except:
                pass

    papers = pd.DataFrame({'title': titles, 'abstract': abstracts})
    papers = papers.dropna()
    papers["title"] = papers["title"].apply(lambda x: re.sub('\s+', ' ', x))
    papers["abstract"] = papers["abstract"].apply(lambda x: re.sub('\s+', ' ', x))

    del titles, abstracts
    papers.to_csv("data/papers.csv", index=False)
    return papers


if __name__ == '__main__':
    """For training uncomment"""
    # news_df = pd.read_csv("data/papers.csv")
    # model = get_base_model()
    # train(news_df=news_df, base_model=model)
    """For generation of text for given title in predict function"""
    predict()
    """For building dataset from JSON snapshot file"""
    # build_dataset()
