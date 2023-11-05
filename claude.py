
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


"""

You are a helpful tool helping a user categorize and summarize their web browsing. 
The following are summaries for web pages in the Learning about RAGS category. Create a 3-5 sentence high level summary of all the summaries. At the end, provide a suggestion on what the user should do next that is helpful and actionable with respect to their intention with these web pages


Category: learning more about RAGS
Site 1:
https://research.ibm.com/blog/retrieval-augmented-generation-RAG
What is retrieval-augmented generation?

RAG is an AI framework that grounds large language models on accurate, up-to-date facts retrieved from knowledge bases, improving response quality and providing transparency into the models' reasoning. RAG reduces the need to constantly retrain models on new data, lowers computational costs, and helps mitigate risks like hallucinated or biased responses. Implementing RAG involves innovations in retrieval to identify relevant facts and generation to optimally structure those facts when querying the model.

Site 2:
https://www.promptingguide.ai/techniques/rag
Retrieval Augmented Generation (RAG)
Retrieval augmented generation (RAG) combines a neural information retrieval system with a text generation model to improve language model performance on knowledge-intensive tasks. RAG retrieves relevant context documents given an input, concatenates them to the input prompt, and feeds them to the text generator to produce an output. Recent work shows RAG improves factual correctness and reliability of language model outputs without full model retraining.

Site 3:
https://gpt-index.readthedocs.io/en/latest/getting_started/concepts.html
High-Level Concepts
The web page provides an overview of key concepts and stages involved in building language models using Retrieval Augmented Generation (RAG), including loading data, indexing it, storing indexes, querying data, and evaluating model performance. It introduces important techniques like using connectors to ingest data, generating vector embeddings, retrieving relevant contexts, and synthesizing responses. The page groups common use cases into 3 categories - query engines, chat engines, and agents.

Site 4:
https://colab.research.google.com/drive/1TAOHDqSQY4uL-ZDdStOR4db4nmB4kIMu?usp=sharing#scrollTo=tO91veANW2XY
Building RAG with Together API and Atlas Vector Search

This tutorial demonstrates how to build a Retrieval Augmented Generation (RAG) system using Together API and MongoDB Atlas. It shows how to embed documents from a sample movie dataset into vector representations using Together Embeddings API, index them in Atlas, and perform semantic search to retrieve relevant documents. The retrieved documents are then used to augment a prompt which is fed into a Together generative model to produce a contextualized output. Overall, it provides a complete guide on constructing an end-to-end RAG system leveraging Together and Atlas to improve generative AI applications.
"""

claude_key = "ADD_API_KEY"


category = "Learning about RAGS"
example_site_summaries = [
    {
        "url": "https://research.ibm.com/blog/retrieval-augmented-generation-RAG",
        "title": "What is retrieval-augmented generation?",
        "summary": "RAG is an AI framework that enhances large language models by grounding them on accurate, up-to-date facts from knowledge bases. It enhances response quality, provides reasoning transparency, reduces retraining frequency, lowers computational costs, and mitigates issues like biased responses. It involves innovative retrieval of facts and their optimal structuring during model queries."
    },
    {
        "url": "https://www.promptingguide.ai/techniques/rag",
        "title": "Retrieval Augmented Generation (RAG)",
        "summary": "Retrieval augmented generation (RAG) merges a neural information retrieval system with a text generation model to boost performance on knowledge-intensive tasks. It enhances factual correctness and reliability of outputs by retrieving context documents based on the input, concatenating them, and using this enriched input to inform the text generator, all without the need for full model retraining."
    },
    {
        "url": "https://gpt-index.readthedocs.io/en/latest/getting_started/concepts.html",
        "title": "High-Level Concepts",
        "summary": "This source outlines key concepts and processes in developing language models with RAG, such as data loading, indexing, storing, querying, and performance evaluation. It details methods like data ingestion connectors, vector embeddings generation, context retrieval, and response synthesis, categorizing common use cases into query engines, chat engines, and agents."
    },
    {
        "url": "https://colab.research.google.com/drive/1TAOHDqSQY4uL-ZDdStOR4db4nmB4kIMu?usp=sharing",
        "title": "Building RAG with Together API and Atlas Vector Search",
        "summary": "This tutorial offers a step-by-step guide to constructing an RAG system with Together API and MongoDB Atlas. It explains embedding documents into vector representations, indexing them in Atlas, and executing semantic searches to retrieve relevant documents. These documents augment prompts for a generative model to create contextual outputs, illustrating a comprehensive approach to developing RAG for generative AI applications."
    }
]

example_samples = [
    {
        "id": 1,
        "full_text": "Some random text of no use"
    },
    {
        "id": 2, 
        "full_text": "Some random text of no use"
    },
    {
        "id": 3,
        "full_text": "Some random text of no use"
    },
    {
        "id": 4,
        "full_text": "Some random text of no use"
    },
]


def generate_categories_prompt(samples):
    prefix = f"You are a helpful tool helping a user categorize and summarize their web browsing. Create categories for the following web pages."

    suffix = f"**Under each category list the ids of the web pages that fit into each category**"

    summary_list = [f"Website {site["id"]}:\nPage content:{site["full_text"]}" for site in samples]

    prompt = prefix + "\n\n" + "\n\n".join(summary_list) + "\n\n" + suffix

    return prompt

def get_generate_categories(anthropic, samples):
    completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=4000,
    prompt=f"{HUMAN_PROMPT} {generate_categories_prompt(samples)}{AI_PROMPT}",
    )
    return completion.completion


def category_summary_prompt(category, site_summaries):

    prefix = f"You are a helpful tool helping a user categorize and summarize their web browsing. The following are summaries for web pages in the {category} category. Create a 3-5 sentence high level summary of all the summaries." #At the end, provide a suggestion on what the user should do next that is helpful and actionable with respect to their intention with these web pages"

    suffix = f"**Provide your response in a markdown code block with hyperlinks. Create inline hyperlinks to the urls of the web page when relevant, and bold or italicize when useful in conveying information.**"

    summary_list = [f"Site {i+1}:\n{site["url"]}\n{site["title"]}\n{site["summary"]}" for i, site in enumerate(site_summaries)]

    prompt = prefix + "\n\n" + "\n\n".join(summary_list) + "\n\n" + suffix
    
    return prompt



def get_category_summary_completion(anthropic, category, site_summaries):
    completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=1000,
    prompt=f"{HUMAN_PROMPT} {category_summary_prompt(category, site_summaries)}{AI_PROMPT}",
    )
    return completion.completion







client = Anthropic(api_key=claude_key)

# output = get_category_summary_completion(client, category, example_site_summaries)

output = get_generate_categories(client, example_samples)

print(output)


# Based on the following web history, give 3 possibilities for what the user could be doing




# Based on the user's intention {intention}, categorize the following web pages into actionable categories and provide 3
# ideal next steps to further their intention





