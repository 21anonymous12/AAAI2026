# MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering

## Abstract
Recent advances in Large Language Models (LLMs) have significantly improved table understanding tasks such as Table Question Answering (TableQA), yet challenges remain in ensuring reliability, scalability, and efficiency, especially in resource-constrained or privacy-sensitive environments. In this paper, we introduce MATA, a multi-agent TableQA framework that leverages multiple complementary reasoning paths and a set of tools built with small language models. MATA generates candidate answers through diverse reasoning styles for a given table and question, then refines or selects the optimal answer with the help of these tools. Furthermore, it incorporates an algorithm designed to minimize expensive LLM agent calls, enhancing overall efficiency. MATA maintains strong performance with small, open-source models and adapts easily across various LLM types. Extensive experiments on two benchmarks of varying difficulty with ten different LLMs demonstrate that MATA achieves state-of-the-art accuracy and highly efficient reasoning while avoiding excessive LLM inference. Our results highlight that careful orchestration of multiple reasoning pathways yields scalable and reliable TableQA. 

### This page is for anonymous submissions to AAAI 2026.

Here, you can find the experimental code, and fine-tuned model checkpoints for MATA, which we have developed for our research.


---
## MATA scheduler Checkpoint
You can download the MATA scheduler checkpoint from the following [link](https://drive.google.com/file/d/1aqeB_05z8MMWXYn_kiKoTFVlCfI__Meb/view?usp=sharing).

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
**3. Place [the scheduler checkpoint](https://drive.google.com/file/d/1aqeB_05z8MMWXYn_kiKoTFVlCfI__Meb/view?usp=sharing) inside the [`scheduler` folder](https://github.com/21anonymous12/AAAI2026/tree/main/scheduler).**

**4. Run the following code.**
```bash
ollama serve
```
**5. Check whether the model you want to use is supported by Ollama on the [official Ollama website](https://ollama.com/search), then pull the corresponding model using the code below. (The model name `phi4:14b` in the code is just an example.)**
```bash
ollama pull phi4:14b
```

**6. If you want to change the model, you need to modify the code in the following four locations:**

  * Line 56 in `MATA.py`

  * Line 25 in `adjust_context.py` inside the `utils` folder

  * The `model_name` variable on line 4 in `adjust_context.py` inside the `utils` folder: this loads the tokenizer for your chosen model from [Hugging Face](https://huggingface.co/)

  * The `max_context` variable on line 4 in `adjust_context.py` inside the `utils` folder: this sets the maximum context length supported by your chosen model


**7. Our code was developed in an [Anaconda](https://www.anaconda.com/) environment. Please run the code below to create a new virtual environment. This will make it easy to install the libraries required for MATA.**
```bash
conda env create -f ./langchain.yml
```

**8. Download the scheduler checkpoint from the following [link](https://drive.google.com/file/d/1aqeB_05z8MMWXYn_kiKoTFVlCfI__Meb/view?usp=sharing) and place it inside the `scheduler` folder.**

**9. Run the following code.**
```bash
python MATA.py --config config.yaml
```

**10. If you do not want to use the scheduler or want to increase the number of self-refinement iterations, you can either modify the `config.yaml` file or run the code as shown below.**
```bash
python MATA.py --config config.yaml --Use_Scheduler False --N 5
```

**Notes:** This repository provides code for using MATA with the `phi4:14b` model. If you want to use a different model, please follow the guidelines mentioned above.

--- 

### Experiments codes

You can find the codes for MATA and the baselines used in the experiments at the following [link](https://github.com/21anonymous12/AAAI2026/tree/main/experiments).
