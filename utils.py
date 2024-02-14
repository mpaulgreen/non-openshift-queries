from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pprint import pprint
from langchain_openai import ChatOpenAI
import os
from keybert.llm import TextGeneration
from keybert import KeyLLM, KeyBERT
from langchain_community.document_loaders import PyPDFLoader


def writeToFile(filename:str, responses: list, mode: str):
    with open(r'./'+filename, mode) as fp:
        for response in responses:
            fp.write(response['query']+ ' '+response['text'] + "\n")
    fp.close()

def questions(filename: str) -> list:
    eval_questions = []
    with open('./'+ filename, 'r') as file:
        for line in file:
            item = line.strip()
            # print(item)
            eval_questions.append(item)
    return eval_questions


def evaluate(template: str,llm: ChatOpenAI, queries: list) -> (list, list):
    prompt_instructions = PromptTemplate.from_template(
            template
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_instructions, verbose=False)
    invalid_results = []
    valid_results = []
    for query in queries:
        response = llm_chain(inputs={"query": query})
        if "INVALID" in response["text"]:
            invalid_results.append(response)
            continue
        elif "VALID" in response["text"]:
            valid_results.append(response)
        else:
            invalid_results.append(response)

    return (valid_results, invalid_results)

def getDocs(doc) -> list:
    # print(f"""Generating keywords for {doc}""")
    loader = PyPDFLoader(doc)
    pages = loader.load_and_split()
    documents = []
    for page in pages:
        documents.append(page.page_content)
    return documents


def common_member(a, b):   
    a = [x.upper() for x in a]
    b = [x.upper() for x in b]
    a_set = set(a)
    b_set = set(b)
    # print(a_set.intersection(b_set))
    # check length 
    if len(a_set.intersection(b_set)) > 0:
        return(a_set.intersection(b_set))  
    else:
        return None
    

def evaluate_keyword_search(filename, queries, vocabs, kw_model):
    for query in queries:
        # print(query)
        keywords = kw_model.extract_keywords(query)
        query_keywords = []
        for keyword in keywords:
            if len(keyword) > 0:
                query_keywords.append(keyword[0])
        with open(r'./outputs/'+ filename, 'a') as fp:
            if common_member(query_keywords, vocabs) is not None:
                fp.write(query+"VALID"+ "\n") 
            else:
                fp.write(query+"INVALID"+ "\n") 

        fp.close() 
        
def evaluateparesponse(template: str,llm: ChatOpenAI, queries: list) -> list:
    prompt_instructions = PromptTemplate.from_template(
            template
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_instructions, verbose=False)
    valid_results = []
    for query in queries:
        response = llm_chain(inputs={"query": query})
        valid_results.append(response)
    return valid_results