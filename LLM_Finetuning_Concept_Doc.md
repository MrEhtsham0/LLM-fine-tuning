# LLM Fine-Tuning Concept Doc

This repository is structured like a learning path, starting with core ML/NLP foundations and moving toward modern LLM fine-tuning, alignment, optimization, and multimodal training.

## 1. Beginner Foundations

These notebooks build the basic skills needed before working with LLMs.

- `Basics of Fine tuning/transformers.ipynb`
  - PyTorch tensors
  - CPU vs GPU
  - Basic tensor operations
  - Data loading and device placement

- `Basics of Fine tuning/huggingface_crash_course.ipynb`
  - Hugging Face login methods
  - Hub authentication
  - Dataset handling
  - Working with hosted models and tokens

- `Basics of Fine tuning/BERT_Finetuning.ipynb`
  - Text classification with BERT
  - Tokenization and preprocessing
  - Supervised training loops
  - Common NLP use cases like sentiment and intent classification

- `Basics of Fine tuning/Knowledge_DIstillation_in_Deep_Learning.ipynb`
  - Teacher-student training
  - Soft targets and temperature scaling
  - Compressing large models into smaller ones

- `Basics of Fine tuning/Why_finetuning_was_challanging_in_LSTM (1).ipynb`
  - LSTM-based sentiment classification
  - Sequence padding
  - Reusing trained models
  - Why older architectures made fine-tuning harder than modern transformers

## 2. Core LLM Fine-Tuning

This is the main shift from classical NLP into modern LLM adaptation.

- `LLM Fine-Tuning-14-Train-LLMs-on-Your-PDF-Text-Data -Domain-Specific-Fine-Tuning-with-HuggingFace/`
  - Domain-specific pretraining on plain text
  - Building training data from PDFs or text corpora
  - Non-instruction fine-tuning

- `LLM Fine-Tuning-15-Instruction Fine-Tuning Explained -Domain-Specific Fine-Tuning with Hugging Face/Instruction_finetuning_on_domain_specific_dataset.ipynb`
  - Instruction tuning
  - Prompt-response style datasets
  - Turning a base model into a task-following model

- `LLM Finetuning-Crash-Course/Final_Finetuning_all_in_one.ipynb`
  - A full pipeline view
  - Base model -> non-instruction model -> instruction model -> preference model
  - LoRA-based fine-tuning with Hugging Face / PEFT

## 3. Alignment and Preference Training

After instruction tuning, the next step is to align model behavior with human or preference signals.

- `LLM Fine-Tuning-16-Preference-based-training/Preference_Aligned_Training_DPO_final.ipynb`
  - Preference learning
  - Direct Preference Optimization (DPO)
  - Ranking better and worse answers
  - Aligning model outputs with chosen responses

## 4. Efficiency and Deployment

These notebooks focus on making models smaller, faster, and more practical to run.

- `LLM Fine-Tuning-12-13-LLM-Quantization/LLM_Quantization/*`
  - GPTQ
  - AWQ
  - GGUF / GGML
  - Model quantization basics

- `LLM Fine-Tuning-12-13-LLM-Quantization/LLM-Quantization-Part-2/*`
  - More quantization examples
  - QAT in LLMs
  - Practical compression workflows

Main idea:

- Reduce memory usage
- Improve inference speed
- Make deployment possible on limited hardware

## 5. Training Frameworks and Tooling

These notebooks show how to fine-tune with common production-oriented tools.

- `LLM Fine-Tuning-17-Llama-Factory/llamafactory.ipynb`
  - Using LLaMA-Factory
  - Dataset setup and training workflow

- `LLM Fine-Tuning-18-unsloth/unsloth_practical.ipynb`
  - Fast fine-tuning with Unsloth
  - Memory-efficient workflows

- `LLM Fine-Tuning-19-Axolotl/axolotl_final_code.ipynb`
  - Axolotl training pipeline
  - Config-driven fine-tuning setup

- `LLM Fine-Tuning-unsloth-vs-hf/`
  - Comparing Unsloth and Hugging Face approaches
  - Practical tradeoffs for speed and simplicity

## 6. Provider-Specific Fine-Tuning

These notebooks show hosted or closed-model fine-tuning workflows.

- `LLM Fine-Tuning-20-GPT-Finetuning/openai_api_and_finetuning_of_gpt_model.ipynb`
  - OpenAI API usage
  - JSONL training data
  - Fine-tuning workflow for GPT models

- `LLM Fine-Tuning-21-GEMINI-Finetuning/gemini_finetuning_clean.ipynb`
  - Gemini fine-tuning workflow
  - Google GenAI / Vertex AI style setup
  - Tuning jobs and evaluation concepts

## 7. Specialized Model Types

These notebooks move beyond standard text-only fine-tuning.

- `LLM Fine-Tuning-22-Finetune-Any-SLM/finetune_any_SLM.ipynb`
  - Fine-tuning smaller language models
  - General SLM adaptation ideas

- `LLM Fine-Tuning-23-Multimodal-LLM-Finetuning/Vision_Model_Finetuning.ipynb`
  - Vision-language fine-tuning
  - Multimodal adapters and LoRA-style training
  - Image + text training workflows

- `LLM Fine-Tuning-23-Multimodal-LLM-Finetuning/image_data_builder.ipynb`
  - Preparing multimodal datasets
  - Data formatting for vision tasks

- `LLM Fine-Tuning-24-Embedding-and-Embedding-Finetuning/Embedding_FT.ipynb`
  - Embedding models
  - Retrieval quality
  - Similarity search and vector-based workflows

## 8. Related But Separate: Advanced RAG

The `Advanced_RAG/` folder is not fine-tuning itself, but it is closely related to LLM systems.

- `01_Introduction_To_RAG.ipynb`
  - Basic RAG architecture
- `02_Query_Transformations.ipynb`
  - Reformulating user queries
- `03_Routing_To_Datasources.ipynb`
  - Choosing the correct data source
- `04_Indexing_To_VectorDBs.ipynb`
  - Indexing strategies and vector databases
- `05_Retrieval_Mechanisms.ipynb`
  - Reranking and fusion techniques
- `06_Self_Reflection_Rag.ipynb`
  - Self-grading and reflection
- `07_Agentic_Rag.ipynb`
  - Agent-based retrieval and generation flows
- `08_Adaptive_Rag_Agent.ipynb`
  - Adaptive routing and retrieval behavior
- `09_Corrective_RAG_Agent.ipynb`
  - Corrective feedback loops
- `10_LLAMA_3_Rag_Agent_Local.ipynb`
  - Local Llama 3 RAG setup

## 9. Suggested Learning Order

1. Start with PyTorch, tensors, and Hugging Face basics.
2. Learn BERT fine-tuning and classic NLP classification.
3. Understand knowledge distillation and older sequence models like LSTM.
4. Move into domain-specific LLM pretraining.
5. Learn instruction fine-tuning.
6. Study preference-based alignment like DPO.
7. Learn quantization and memory-efficient deployment.
8. Practice with tools like Unsloth, Axolotl, and LLaMA-Factory.
9. Explore multimodal fine-tuning and embedding tuning.
10. Use Advanced RAG to combine retrieval with generation.

## 10. One-Line Summary

This repository covers the full spectrum from basic NLP and PyTorch foundations to instruction tuning, alignment, quantization, multimodal fine-tuning, embedding optimization, and advanced retrieval-augmented generation.