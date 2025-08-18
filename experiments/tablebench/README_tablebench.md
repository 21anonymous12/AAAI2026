# How to Reproduce MATA Experiment Results on the Tablebench Dataset

## TableBench

[TableBench](https://tablebench.github.io/) is a comprehensive benchmark for complex table QA, spanning 18 subcategories across fact checking, numerical reasoning, data analysis, and visualization. Tables are sourced from diverse domains such as finance, sports, and science. The benchmark emphasizes real-world complexity and supports multiple reasoning paradigms including TCoT, SCoT, and PoT.  In our experiments, as our focus is on entity-type answers, we excluded table-to-text samples from TableBench, which provide sentence-level ground truth. Therefore, 693 out of 886 TableBench test samples are included in our evaluation.


By following the procedure below, you can reproduce the experimental results on the Tablebench dataset, which is one of the benchmark datasets used in our paper.

---


## MATA scheduler Checkpoint
You can download the MATA scheduler checkpoint from the following [link](https://drive.google.com/file/d/1034behq_VONXuJOlvCKuFRXNYkmNERTI/view?usp=sharing).

## MATA confidence checker Checkpoint
You can download the MATA confidence checker checkpoint from the following [link](https://huggingface.co/7anonymous7/confidence_checker).

---
## How to Use

**1. Clone this repository using the web URL.**
```bash
git clone https://github.com/21anonymous12/AAAI2026.git
```
**2. To use MATA, you need to install [Ollama](https://ollama.com/). Please run the following code in your local environment. Our code is designed to be used on Linux systems.**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
**3. Place [the scheduler checkpoint](https://drive.google.com/file/d/1034behq_VONXuJOlvCKuFRXNYkmNERTI/view?usp=sharing) inside the [`scheduler` folder](https://github.com/21anonymous12/AAAI2026/tree/main/scheduler).**

**4. Run the following code.**
```bash
ollama serve
```
**5. Check whether the model you want to use is supported by Ollama on the [official Ollama website](https://ollama.com/search), then pull the corresponding model using the code below. (The model name `qwen2.5:32b-instruct` in the code is just an example.)**
```bash
ollama pull qwen2.5:32b-instruct
```

**6. Move the `MATA_tablebench.py` file and the `Tablebench_loader.py` file to the root folder.**

**7. If you want to change the model, you need to modify the code in the following four locations:**

  * Line 130, 223, and 269 in `MATA_tablebench.py`

  * Line 25 in `adjust_context.py` inside the `utils` folder

  * The `model_name` variable on line 4 in `adjust_context.py` inside the `utils` folder: this loads the tokenizer for your chosen model from [Hugging Face](https://huggingface.co/)

  * The `max_context` variable on line 4 in `adjust_context.py` inside the `utils` folder: this sets the maximum context length supported by your chosen model


**8. Our code was developed in an [Anaconda](https://www.anaconda.com/) environment. Please run the code below to create a new virtual environment. This will make it easy to install the libraries required for MATA.**
```bash
conda env create -f ./langchain.yml
```

**9. Download the scheduler checkpoint from the following [link](https://drive.google.com/file/d/1034behq_VONXuJOlvCKuFRXNYkmNERTI/view?usp=sharing) and place it inside the `scheduler` folder.**


**10. Run the following code.**
```bash
python MATA_tablebench.py --config config.yaml
```

**11. If you do not want to use the scheduler or want to increase the number of self-refinement iterations, you can either modify the `config.yaml` file or run the code as shown below.**
```bash
python MATA_tablebench.py --config config.yaml --Use_Scheduler False --N 5
```

**Notes:** This repository provides code for using MATA with the `qwen2.5:32b-instruct` model. If you want to use a different model, please follow the guidelines mentioned above.

