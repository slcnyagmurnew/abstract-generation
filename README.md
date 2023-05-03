## Abstract Generation using Article Title with GPT-2

This study is inspired by https://www.modeldifferently.com/en/2021/12/generaci%C3%B3n-de-fake-news-con-gpt-2/ 

Base dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv (download and add file to **/data** folder)

### About

Essentially, in this study (an attempt to try GPT model), new article abstracts about the given categories (AI, ML etc.) are generated based on the given headings with using GPT-2 model.

There are 3 stages:

**1. Build Dataset:** Pandas dataframe and papers.csv file are created using **build_dataset** function. <br>
**2. Train Model:** papers.csv file is read and GPT-2 model is created before **train** function is called. <br>
**3. Prediction:** Title will be used for generation of abstract is saved into **predict** function.

PyTorch is using GPU for both training and prediction operations. However, they would be too slow when CPU is used instead.

