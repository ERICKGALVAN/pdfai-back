# pdf_chat_backend

Este es el backend del proyecto de tesina "Comparación cualitativa y cuantitativa de respuestas generadas por Modelos de Lenguaje Pre Entrenados" el cual busca comparar las respuestas y textos generados por ciertos modelos elegidos y determinar las características que los hacen mejores o peores.

[Proyecto frontend](https://github.com/ERICKGALVAN/pdf-chat?tab=readme-ov-file)

## Modelos utilizados

- OpenAI GPT 3.5 Turbo: [OpenAI/Gpt3.5](https://platform.openai.com/docs/models/gpt-3-5-turbo)
- MistralAI Mistral-7B-Instruct-v0.2: [HuggingFace/Mistral7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Google Flan T5 Base: [HuggingFace/FlanT5](https://huggingface.co/google/flan-t5-base)  
  @misc{https://doi.org/10.48550/arxiv.2210.11416,
  doi = {10.48550/ARXIV.2210.11416},
  url = {https://arxiv.org/abs/2210.11416},
  author = {Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Eric and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and Webson, Albert and Gu, Shixiang Shane and Dai, Zhuyun and Suzgun, Mirac and Chen, Xinyun and Chowdhery, Aakanksha and Narang, Sharan and Mishra, Gaurav and Yu, Adams and Zhao, Vincent and Huang, Yanping and Dai, Andrew and Yu, Hongkun and Petrov, Slav and Chi, Ed H. and Dean, Jeff and Devlin, Jacob and Roberts, Adam and Zhou, Denny and Le, Quoc V. and Wei, Jason},
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Scaling Instruction-Finetuned Language Models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
  }

## Las métricas cuantitativas utilizadas fueron probadas con los siguientes modelos:

- BLEU: [HuggingFace/BLEU](https://huggingface.co/spaces/evaluate-metric/bleu)  
  @INPROCEEDINGS{Papineni02bleu:a,
  author = {Kishore Papineni and Salim Roukos and Todd Ward and Wei-jing Zhu},
  title = {BLEU: a Method for Automatic Evaluation of Machine Translation},
  booktitle = {},
  year = {2002},
  pages = {311--318}
  }
  @inproceedings{lin-och-2004-orange,
  title = "{ORANGE}: a Method for Evaluating Automatic Evaluation Metrics for Machine Translation",
  author = "Lin, Chin-Yew and
  Och, Franz Josef",
  booktitle = "{COLING} 2004: Proceedings of the 20th International Conference on Computational Linguistics",
  month = "aug 23{--}aug 27",
  year = "2004",
  address = "Geneva, Switzerland",
  publisher = "COLING",
  url = "https://www.aclweb.org/anthology/C04-1072",
  pages = "501--507",
  }

- BERTSCORE: [HuggingFace/BERTSCORE](https://huggingface.co/spaces/evaluate-metric/bertscore)  
  @inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu\* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
  }

- ROUGE: [HuggingFace/ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge)  
  @inproceedings{lin-2004-rouge,
  title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
  author = "Lin, Chin-Yew",
  booktitle = "Text Summarization Branches Out",
  month = jul,
  year = "2004",
  address = "Barcelona, Spain",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/W04-1013",
  pages = "74--81",
  }

- Wiki Split: [HuggingFace/WikiSplit](https://huggingface.co/spaces/evaluate-metric/wiki_split)  
  @article{rothe2020leveraging,
  title={Leveraging pre-trained checkpoints for sequence generation tasks},
  author={Rothe, Sascha and Narayan, Shashi and Severyn, Aliaksei},
  journal={Transactions of the Association for Computational Linguistics},
  volume={8},
  pages={264--280},
  year={2020},
  publisher={MIT Press}
  }
