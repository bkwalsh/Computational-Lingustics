# A Deep Learning Approach to Abusive Language and Hate Speech Detection for the Javanese Language

## Abstract

This paper develops a deep learning approach to abusive language and hate speech detection using Javanese and Indonesian large language models (LLMs). We experiment on a Javanese Twitter dataset created by [Putri et al.](http://www.wcse.org/WCSE_2021_Spring/011.pdf), aiming to beat their best F-measure of 0.780. Using a fine-tuned Javanese GPT-2 as a feature extractor for our classifier, the model achieves an F-measure of 0.811. Surprisingly, utilizing an Indonesian GPT-2 as the feature extractor yields a superior F-measure 0.854, potentially attributable to code-mixing in Javanese Twitter data or the model’s training on colloquial language. This study further explores the nuances
of hate speech detection in Javanese, emphasizing language and model choice.

<img src="https://github.com/KevinyWu/KevinyWu/blob/main/images/javanese_hate_speech.png" alt="drawing" width="450"/>

Please see our [paper](https://github.com/KevinyWu/javanese-hate-speech/blob/main/A_Deep_Learning_Approach_to_Abusive_Language_and_Hate_Speech_Detection_for_the_Javanese_Language.pdf).

## Code

To run the code please follow the instructions:

1. Clone the repository
2. Install the requirements in `requirements.txt`
3. Run `data_preparation.ipynb` to clean and split the data
4. Run `javanese_experiments.ipynb` to train and evaluate the models (GPU is recommended)
5. See `model_analysis.ipynb` for further analysis of the best model, Indonesian GPT-2
