# MIAS: Multi-Inference Answer Selection for Reliable and Flexible Table Understanding

## Abstract
Recent advances in Large Language Models (LLMs) have significantly improved table understanding tasks such as Table Question Answering (TableQA), yet challenges remain in ensuring reliability, scalability, and efficiency, especially in resource-constrained environments. In this paper, we introduce MIAS (Multi-Inference Answer Selection), a modular TableQA framework that integrates three complementary reasoning paths: Chain-of-Thought (CoT), Program-of-Thought (PoT), and text-to-SQL. MIAS employs a lightweight selector to robustly identify the most accurate answer among these paths, incorporating built-in hallucination detection without additional LLM calls. An optional scheduler further enhances efficiency by predicting the most promising reasoning paths, reducing redundant inference. Unlike prior work that depends heavily on closed-source LLMs, MIAS maintains strong performance with small, open-source models and adopts easily across various LLM types. Extensive experiments on five widely-used benchmarks demonstrate that MIAS achieves state-of-the-art accuracy, efficient inference, and superior hallucination detection.

### This page is for anonymous submissions to EMNLP 2025.

Here, you can find the experimental code, and fine-tuned model checkpoints for MIAS, which we have developed for our research.


---
## MIAS scheduler Checkpoint
You can download the MIAS scheduler checkpoint from the following [link](https://drive.google.com/file/d/1034behq_VONXuJOlvCKuFRXNYkmNERTI/view?usp=sharing).

## MIAS selector Checkpoint
You can download the MIAS selector checkpoint from the following [link](https://huggingface.co/7anonymous7/MIAS_selector).

---
## How to Use

**1. Clone this repository using the web URL.**
```bash
git clone https://github.com/21anonymous12/EMNLP..git
```
**2. To use MIAS, you need to install [Ollama](https://ollama.com/). Please run the following code in your local environment. Our code is designed to be used on Linux systems.**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
**3. Place [the scheduler checkpoint](https://drive.google.com/file/d/1034behq_VONXuJOlvCKuFRXNYkmNERTI/view?usp=sharing) inside the [`scheduler` folder](https://github.com/21anonymous12/EMNLP./tree/main/scheduler).**

**4. Run the following code.**
```bash
ollama serve
```
**5. Check whether the model you want to use is supported by Ollama on the [official Ollama website](https://ollama.com/search), then pull the corresponding model using the code below. (The model name `phi4:14b` in the code is just an example.)**
```bash
ollama pull phi4:14b
```

**6. If you want to change the model, you need to modify the code in the following four locations:**

  * Line 56 in `MIAS.py`

  * Line 25 in `adjust_context.py` inside the `utils` folder

  * The `model_name` variable on line 4 in `adjust_context.py` inside the `utils` folder: this loads the tokenizer for your chosen model from [Hugging Face](https://huggingface.co/)

  * The `max_context` variable on line 4 in `adjust_context.py` inside the `utils` folder: this sets the maximum context length supported by your chosen model


**7. Our code was developed in an [Anaconda](https://www.anaconda.com/) environment. Please run the code below to create a new virtual environment. This will make it easy to install the libraries required for MIAS.**
```bash
conda env create -f ./langchain.yml
```

**8. Download the scheduler checkpoint from the following [link](https://drive.google.com/file/d/1034behq_VONXuJOlvCKuFRXNYkmNERTI/view?usp=sharing) and place it inside the `scheduler` folder.**

**9. Run the following code.**
```bash
python MIAS.py --config config.yaml
```

**10. If you do not want to use the scheduler or want to increase the number of self-refinement iterations, you can either modify the `config.yaml` file or run the code as shown below.**
```bash
python MIAS.py --config config.yaml --Use_Scheduler False --N 5
```

**Notes:** This repository provides code for using MIAS with the `phi4:14b` model. If you want to use a different model, please follow the guidelines mentioned above.

--- 

### Experiments codes

You can find the codes for MIAS and the baselines used in the experiments at the following [link](https://github.com/21anonymous12/EMNLP./tree/main/experiments).
