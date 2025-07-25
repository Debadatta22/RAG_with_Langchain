# RAG_with_Langchain
Retrieval Augmented Generation (RAG) with LangChain" lab experience hosted via IBM SkillsBuild and IBM Cloud

### Click here

<p align="center">
  <a href="https://colab.research.google.com/github/ibm-granite-community/granite-snack-cookbook/blob/main/recipes/RAG/RAG_with_Langchain.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
  </a>
</p>


Internship Via Edunet Foundations

<img width="1334" height="585" alt="image" src="https://github.com/user-attachments/assets/da82ea09-88bc-4087-b130-47ce08aa13af" />

<img width="1363" height="597" alt="image" src="https://github.com/user-attachments/assets/9f6d3979-9a6e-467c-a624-86fc46866d87" />

<img width="1239" height="595" alt="image" src="https://github.com/user-attachments/assets/27102471-cfc2-4078-a0ce-2c2c7344bf35" />


<img width="1365" height="756" alt="image" src="https://github.com/user-attachments/assets/843daf5b-5f67-42ad-b992-63e56e36b97c" />





## 🔍 Introduction
In the modern era of generative AI, language models such as GPT, BERT, and LLaMA have revolutionized natural language processing by generating human-like responses. However, these models are limited by the data they were trained on, leading to knowledge gaps, outdated information, or hallucinated outputs.

To overcome this, the Retrieval-Augmented Generation (RAG) architecture was introduced — a hybrid approach that combines retrieval-based learning with generative models. This lab, offered via IBM SkillsBuild, provided a hands-on opportunity to build and explore a RAG-based pipeline using LangChain, a Python framework that simplifies working with LLM-powered applications.

In this lab, I implemented Retrieval-Augmented Generation (RAG) using the LangChain framework in combination with IBM Granite AI Models, hosted and orchestrated via IBM Cloud resources. RAG is a cutting-edge architecture designed to enhance the capabilities of large language models (LLMs) by integrating real-time retrieval of relevant information from external knowledge bases. This significantly improves the factual accuracy and domain-specific relevance of AI responses.



## 🎯 Objective of the Lab
The goal of the "RAG with LangChain" lab was to:

Understand how RAG enhances LLMs with real-time information.

Build a functional RAG pipeline using LangChain.

Learn how to retrieve relevant knowledge from a dataset.

Augment user queries with contextual documents before generating responses.

Deploy and test the solution on Google Colab using IBM Cloud infrastructure.

## 🛠️ Step-by-Step Approach
1. Environment Setup

The project was run in a Python 3.10–3.12 environment using Google Colab. All dependencies including LangChain, transformers, Milvus, and Replicate APIs were installed.

2. Embedding Model Selection

We used ibm-granite/granite-embedding-30m-english, a compact Granite embedding model from IBM, to convert textual data into dense vector embeddings.

3. Vector Database Initialization

To store and retrieve these embeddings efficiently, Milvus, a vector database, was configured locally using temporary file storage.

4. Language Model Integration

The Granite large language model (ibm-granite/granite-3.3-8b-instruct) was loaded via the Replicate API to perform inference over combined query + context input.

5. Document Processing
   
A real-world document (President Biden’s 2022 State of the Union address) was:

Downloaded from IBM’s dataset repository

Split into manageable chunks using LangChain.TextSplitter

Embedded and stored in the vector database

6. Semantic Search

Upon receiving a user query (e.g., “What did the president say about Ketanji Brown Jackson?”), the system:

Converted the query into an embedding

Retrieved semantically similar text chunks

Used the language model to generate a final answer with context

7. RAG Chain Construction

A full retrieval-augmented generation pipeline was created by combining:

Retriever (semantic search from vector DB)

Combiner (LLM with context-aware prompting)

LangChain chain (automated pipeline execution)

## 📚 What is RAG (Retrieval Augmented Generation)?
**➤ Definition:**
Retrieval Augmented Generation (RAG) is a neural architecture that combines the power of information retrieval (like search engines) with the fluency of generative language models.

Instead of relying purely on the model's pre-trained knowledge, RAG introduces an external knowledge base (KB) or document store, from which relevant information is fetched and used as context for generating accurate, grounded responses.

## 🔁 RAG Workflow:
User Input (Query): A user submits a query or question.

Retrieval: The system searches a vector database or knowledge base to find semantically relevant documents using dense embeddings.

Augmentation: The retrieved documents are appended to the original query to enrich it with external factual context.

Generation: The LLM (e.g., GPT) processes the augmented query and generates a final, accurate, and context-aware response.

## 🛠️ LangChain & Its Role in RAG
LangChain is a Python framework that allows developers to build LLM-powered applications by integrating models, chains, and tools. In the context of RAG:

LangChain provides modules to handle:

Document loading

Text chunking

Embedding generation

Vector storage (like FAISS, Pinecone)

Retrieval mechanisms

Query augmentation

Model prompting and response generation

It acts as a middleware between data sources and large language models, allowing a modular, scalable, and maintainable approach to build RAG pipelines.

## ☁️ How IBM Cloud Supports This Lab
The lab was hosted on Google Colab but integrated with services and APIs from IBM Cloud. Here’s how IBM’s infrastructure enhanced the lab:

✅ Key IBM Cloud Integrations:
Watson AI Services: For deploying AI models and managing language-based APIs.

AutoAI & Machine Learning Pipelines: For advanced users to integrate RAG outputs into real-world ML workflows.

IBM Cloud Object Storage: Ideal for hosting and retrieving large document corpora used in retrieval stages.

IBM Watson Discovery (Advanced): A commercial-grade enterprise search engine that can be used in place of FAISS or open-source vector stores.

IBM Cloud ensures enterprise-grade security, reliability, and scalability, making it a go-to platform for deploying production-ready AI/RAG applications.

## 🧠 Lab Approach: Step-by-Step Breakdown
Setup:

Sign in to Google & GitHub

Access Colab Notebook for RAG Lab

Claim credits if required

Vector Embedding:

A knowledge base is loaded and text is split into chunks.

These chunks are converted into dense vectors using embedding models.

Vector Store (Retrieval):

The vectors are stored in a vector database like FAISS.

Given a user query, relevant vectors (text chunks) are retrieved based on semantic similarity.

Augmentation & Generation:

Retrieved documents are fed along with the query into the language model (like OpenAI, HuggingFace, etc.).

The LLM generates a well-informed answer using both its internal knowledge and the retrieved external context.

Testing & Evaluation:

Multiple queries are tested to analyze the responsiveness and factual correctness.

Edge cases and limitations are discussed.

## 🚀 Skills & Tools Gained
📘 LangChain Framework

🔍 Semantic Search & Embedding

🧠 RAG Architecture Design

🗂️ FAISS Vector Database

☁️ IBM Cloud Integration

💡 LLM-Powered Augmented Applications

🐍 Python Programming with Google Colab

📌 Key Takeaways
✅ Understood and implemented the concept of RAG end-to-end using LangChain

✅ Leveraged IBM’s Granite Embeddings & Instruct Models for production-grade AI inference

✅ Created a scalable AI retrieval pipeline from document ingestion to intelligent answer generation

✅ Used Milvus vector store for high-performance similarity search

✅ Applied prompt templating and RAG chaining to enhance the accuracy of responses

## 🌐 Real-World Applications of RAG
Customer Support Automation

Personalized Search Engines

Domain-specific Chatbots

Document QA Systems

Knowledge Mining from Unstructured Text

## ☁️ IBM Cloud & SkillsBuild Lab Support
Thanks to IBM SkillsBuild and Edunet Foundation, this lab gave access to IBM’s state-of-the-art AI infrastructure, real-world datasets, and hands-on tasks designed by domain experts. The course enabled me to explore advanced concepts in retrieval-augmented AI systems with direct application in modern LLM deployments.

## Certification

<img width="851" height="536" alt="image" src="https://github.com/user-attachments/assets/fd362cb9-536f-4027-a15f-e6e937413a11" />


## ✅ Conclusion
This lab was not just about theory — it was about building a cutting-edge AI solution that mirrors what’s happening in enterprise AI today. With the world moving rapidly toward factual, explainable AI, RAG is becoming a must-know pattern for AI engineers, data scientists, and solution architects.

Thanks to IBM SkillsBuild and Edunet Foundation, learners can gain free access to industry-aligned labs and get certified with credible IBM badges and credentials — an excellent way to build portfolios and stand out in the job market.


## 🔗 References

<p align="left">
  <a href="https://github.com/ibm-granite-community" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-IBM_Granite_Community-181717?style=for-the-badge&logo=github" alt="IBM Granite GitHub" />
  </a>
  <a href="https://huggingface.co/settings/tokens" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Token_Settings-FCC624?style=for-the-badge&logo=huggingface" alt="Hugging Face Tokens" />
  </a>
  <a href="https://replicate.com/ibm-granite/granite-speech-3.3-8b/api" target="_blank">
    <img src="https://img.shields.io/badge/Replicate-Granite_3.3_8b_API-000000?style=for-the-badge&logo=replicate" alt="Replicate Granite API" />
  </a>
</p>

<img width="1328" height="354" alt="image" src="https://github.com/user-attachments/assets/2a8c6b1b-d3a7-47ef-840b-999ef7bd59fd" />


<img width="1284" height="579" alt="image" src="https://github.com/user-attachments/assets/159f6e34-3cef-4663-b1fb-f9de7edbad56" />

<img width="1340" height="592" alt="image" src="https://github.com/user-attachments/assets/00227f46-6019-4256-87a3-ac8e9a7022b6" />

