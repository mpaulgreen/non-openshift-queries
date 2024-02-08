```python
## Redacted row
GPT_API_KEY="<<redacted>>"
IBM_API_KEY="<<redacted>>"
```

#### Terminology

```
True - Assumed valid questions for openshift or kubernetes
False - Assumed invalid questions for openshift or kubernetes
Positive - keyword search or LLM has returned a reponse with VALID
Negative - keyword search or LLM has not returned a reponse with VALID
```

- Evaluationis been done only the basis of 4 metrics that is typically used in binary classification:
    - True Positive
    - True Negative
    - False Positive
    - False Negative

## Imports


```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pprint import pprint
from langchain_openai import ChatOpenAI
from genai import Client,Credentials
from genai.extensions.langchain import LangChainInterface
from genai.text.generation import TextGenerationParameters
import os

from utils import (
    writeToFile,
    questions,
    evaluate,
    common_member,
    evaluate_keyword_search,
    evaluateparesponse
)

from keybert import KeyBERT
```

# gpt-3.5-turbo setup


```python
params = {
    'base_url': 'https://api.openai.com/v1', 
    'api_key': GPT_API_KEY, 
    'model': 'gpt-3.5-turbo', 
    'model_kwargs': {
        'top_p': 0.95, 
        'frequency_penalty': 1.03        
     }, 
    'organization': None, 
    'timeout': None, 
    'cache': None, 
    'streaming': True, 
    'temperature': 0.01, # controls hallucination
    'max_tokens': 20, 
    'verbose': False
}
```


```python
chat_llm = ChatOpenAI(**params)
```

## IBM granite-13b-chat-v2


```python
url = "https://bam-api.res.ibm.com"
model = "ibm/granite-13b-chat-v2"
```


```python
creds = Credentials(
    api_key=IBM_API_KEY,
    api_endpoint=url
)
```


```python
bam_params = {
    "decoding_method": "sample",
    "max_new_tokens": 5,
    "min_new_tokens": 1,
    "random_seed": 42,
    "top_k": 10,
    "top_p": 0.95,
    "repetition_penalty": 1.03,
    "temperature": 0.05,
}
```


```python
params = TextGenerationParameters(**bam_params)
```


```python
client = Client(credentials = creds)
```


```python
ibm_llm = LangChainInterface(client=client, model_id=model, parameters=params)
```

# First Step - Keyword Search

#### The keyword search filter, is to reject questions that are not related to openshift. This is very lenient for enhanced user experience.
##### It will prevent queries like 

```
who is lady gaga?
```

##### But will allow
```
What is openshift?
```

#### This is meant for basic filtering only


```python
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
```

    /opt/conda/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()



```python
vocabs = questions("./inputs/openshift-vocab-manual-cleanse.txt")
```

## Case True


```python
queries = questions("./inputs/valid_questions.txt")
```


```python
filename = 'keyword_valid.txt'
```


```python
with open(r'./outputs/' + filename, 'w') as fp:
    pass
fp.close()

```


```python
evaluate_keyword_search(
    filename = filename,
    vocabs = vocabs,
    queries = queries,
    kw_model = kw_model
)
```

#### Inspect the result


```python
outputs = questions('./outputs/'+filename)
```


```python
invalid_outputs = []
valid_outputs = []
for output in outputs:
    if "INVALID" in output:
        print(output)
        invalid_outputs.append(output)
    else:
        valid_outputs.append(output)
```

    What is the main focus of Chapter 2 in the document?INVALID
    What is the purpose of the spec.overrides parameter in the CVO's configuration?INVALID
    What is the recommended step to take if reported issues need support to proceed?INVALID
    What information is provided in the Total vCPU field?INVALID
    What information can be found in the Add-ons tab?INVALID
    What is the significance of Figure 7.1 in the document?INVALID
    What is CUDA and what is its purpose?INVALID
    What is the purpose of the NVIDIA-Certified Systems?INVALID
    How many pages are in the document?INVALID
    What information can be found in the Red Hat Ecosystem Catalog?INVALID



```python
len(queries), len(invalid_outputs), len(valid_outputs)
```




    (1000, 10, 990)



#### True Positive


```python
true_positive = (len(valid_outputs)/len(queries))*100
true_positive
```




    99.0



#### True Negative


```python
true_negative = (len(invalid_outputs)/len(queries))*100
true_negative
```




    1.0



#### Inference, 1% true negative cases could be reduced by enhancing the vocabulary or it is rejected by LLM

## Case False


```python
queries = questions("./inputs/invalid_questions.txt")
```


```python
filename = 'keyword_invalid.txt'
```


```python
with open(r'./outputs/' + filename, 'w') as fp:
    pass
fp.close()
```


```python
evaluate_keyword_search(
    filename = filename,
    vocabs = vocabs,
    queries = queries,
    kw_model = kw_model
)
```

#### Inspect the result


```python
outputs = questions('./outputs/'+filename)
```


```python
invalid_outputs = []
valid_outputs = []
for output in outputs:
    if "INVALID" in output:
        print(output)
        invalid_outputs.append(output)
    else:
        valid_outputs.append(output)
```

    who is Lady Gaga?INVALID
    give me the best hit of Michael Jackson?INVALID
    How many siblings do you have?INVALID
    Do you have any pets?INVALID
    What is your favorite book?INVALID
    Can you play any musical instruments?INVALID
    Are you a morning person or a night owl?INVALID
    Do you prefer sweet or savory food?INVALID
    Would you rather travel to the future or the past?INVALID
    What is your favorite sport to watch or play?INVALID
    Do you enjoy cooking?INVALID
    What is your all-time favorite movie?INVALID
    Are you a good dancer?INVALID
    Do you have any hidden talents?INVALID
    What is your biggest fear?INVALID
    Do you prefer coffee or tea?INVALID
    Are you a spontaneous or a planner?INVALID
    Have you ever been scuba diving or skydiving?INVALID
    Can you speak any other languages besides English?INVALID
    What is your favorite season?INVALID
    Do you prefer beaches or mountains?INVALID
    Can you ride a bicycle?INVALID
    Can you swim?INVALID
    Are you an introvert or an extrovert?INVALID
    What is your favorite drink?INVALID
    Are you a fan of reality TV shows?INVALID
    Do you enjoy going to concerts or live performances?INVALID
    Can you do any impressions of famous people?INVALID
    Do you prefer to read physical books or e-books?INVALID
    Have you ever been on a road trip? If so, where did you go?INVALID
    Do you have any phobias?INVALID
    Have you ever broken a bone?INVALID
    What is your favorite holiday?INVALID
    Are you a morning shower person or a night shower person?INVALID
    What is your favorite animal?INVALID
    Do you believe in aliens or extraterrestrial life?INVALID
    What is your favorite genre of movies?INVALID
    Can you drive a manual transmission car?INVALID
    Are you a fan of any sports team?INVALID
    Do you enjoy gardening or taking care of plants?INVALID
    Have you ever been to a music festival?INVALID
    What is your favorite TV show or series?INVALID
    Do you have a favorite quote or motto?INVALID
    Can you whistle?INVALID
    Have you ever ridden a horse?INVALID
    Do you enjoy hiking or being outdoors?INVALID
    Are you a fan of poetry?INVALID
    Can you do any magic tricks?INVALID
    Have you ever been in a food eating competition?INVALID
    Are you a fan of spicy food?INVALID
    Can you do any accents or dialects?INVALID
    Have you ever been camping?INVALID
    Do you prefer shopping online or in physical stores?INVALID
    Have you ever been on a cruise?INVALID
    Do you enjoy painting or drawing?INVALID
    Can you solve a Rubik's Cube?INVALID
    Have you ever been in a play or a musical?INVALID
    Are you a fan of comedy shows or stand-up comedy?INVALID
    Have you ever been on a hot air balloon ride?INVALID



```python
len(queries), len(invalid_outputs), len(valid_outputs)
```




    (96, 59, 37)



#### False Positives


```python
false_positive = (len(valid_outputs)/len(queries))*100
false_positive
```




    38.54166666666667



#### False Negatives


```python
false_negative = (len(invalid_outputs)/len(queries))*100
false_negative
```




    61.458333333333336



#### False Positives are high as question like
```
Does Openshift also work as a vacuum cleaner?
```
#### has been passed to allow passing for questions like 
```
What is openshift?
What is a pod?
```

#### But ultimately it is to give an invalid response from a LLM

# SecondStep - with prompts


```python
QUESTION_VALIDATOR_PROMPT_TEMPLATE = """
Instructions:
- You are a question classifying tool and you must answer within provided context
- You are an expert in kubernetes and openshift topics
- Your job is to determine if a question is related to kubernetes or openshift and to provide a one-word response
- If a question is not related to kubernetes or openshift, answer with only the word INVALID
- If a question is related to kubernetes or openshift, answer with the word VALID
- Do not provide explanation, only respond with the chosen words

Example Question:
Can you make me lunch with ham and cheese?
Example Response:
INVALID

Example Question:
Why is the sky blue?
Example Response:
INVALID

Example Question:
Can you help configure my cluster to automatically scale?
Example Response:
VALID

Example Question:
please give me a vertical pod autoscaler configuration to manage my frontend deployment automatically.  Don't update the workload if there are less than 2 pods running.
Example Response:
VALID


Question:
{query}
Context: openshift
Response:
"""
```

## Chat GPT

## Case True


```python
queries = questions('./inputs/valid_questions.txt')
```


```python
valid_response, invalid_response = evaluate(
    template=QUESTION_VALIDATOR_PROMPT_TEMPLATE, 
    llm = chat_llm, 
    queries = queries
)
```

    /opt/conda/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(


#### Inspect the result


```python
writeToFile("./outputs/chatgpt_prompt_valid.txt", valid_response, 'w')
writeToFile("./outputs/chatgpt_prompt_invalid.txt", invalid_response, 'a')
```


```python
len(queries), len(valid_response), len(invalid_response)
```




    (1000, 974, 26)




```python
for response in invalid_response:
    print(response)
```

    {'query': 'How many pages are there in the document mentioned in the context information?', 'text': 'INVALID'}
    {'query': 'What is the file path of the document mentioned in the context information?', 'text': 'INVALID'}
    {'query': 'What are the services that fall under the Kubernetes category on the control plane?', 'text': 'INVALID'}
    {'query': 'What is the primary data store for Kubernetes?', 'text': 'INVALID'}
    {'query': 'What is the role of an NVIDIA GPU driver in vSphere deployments?', 'text': 'INVALID'}
    {'query': 'How many vGPUs can be allocated to worker node VMs in vSphere 7.0?', 'text': 'INVALID'}
    {'query': 'What is the maximum number of vGPUs per VM in vSphere 8.0?', 'text': 'INVALID'}
    {'query': 'What is the maximum number of vGPUs that can be allocated to worker node VMs in vSphere 8.0?', 'text': 'INVALID'}
    {'query': 'What is the purpose of the VIB file in vSphere deployments?', 'text': 'INVALID'}
    {'query': 'What is CUDA and what is its purpose?', 'text': 'INVALID'}
    {'query': 'How does asynchronous processing of operations across different streams improve GPU performance?', 'text': 'INVALID'}
    {'query': 'What is the difference between multi-instance GPU (MIG) and CUDA Multi-Process Service (MPS)?', 'text': 'INVALID'}
    {'query': 'How does CUDA MPS enhance GPU utilization?', 'text': 'INVALID'}
    {'query': 'When would you use MIG instead of utilizing the full power of an entire GPU?', 'text': 'INVALID'}
    {'query': 'How does CUDA Multi-Process Service (MPS) eliminate saturation of GPU compute resources?', 'text': 'INVALID'}
    {'query': 'What is the purpose of the default stream in CUDA?', 'text': 'INVALID'}
    {'query': 'What is the purpose of the Compute Unified Device Architecture (CUDA)?', 'text': 'INVALID'}
    {'query': 'How does CUDA enable general computing on GPUs?', 'text': 'INVALID'}
    {'query': 'What is the benefit of running multiple tasks simultaneously on a GPU?', 'text': 'INVALID'}
    {'query': 'What is the purpose of the MIG feature in the new NVIDIA Ampere architecture?', 'text': 'INVALID'}
    {'query': 'What is the benefit of using vGPU in virtual machines?', 'text': 'INVALID'}
    {'query': 'What is the role of the MIG feature in the NVIDIA Ampere architecture?', 'text': 'INVALID'}
    {'query': 'What are the benefits of using vGPU in virtual machines?', 'text': 'INVALID'}
    {'query': 'What chapter in the document provides an overview of NVIDIA GPU architecture?', 'text': 'INVALID'}
    {'query': 'How many pages are in the document?', 'text': 'INVALID'}
    {'query': 'Why is support for Docker Container Engine deprecated in Kubernetes 1.20?', 'text': 'INVALID'}


#### True Positive


```python
true_positive = (len(valid_response)/1000)*100
true_positive
```




    97.39999999999999



#### True Negative


```python
true_negative = (len(invalid_response)/1000)*100
true_negative
```




    2.6



#### Higher number of true positive proves the framework. Some of the true negatives were due to questions like the following
```
What is CUDA and what is its purpose?
```
#### which are very generic in nature. Fine tuning of the model is required to pass these queries.

## Case False


```python
queries = questions('./inputs/invalid_questions.txt')
```


```python
valid_response, invalid_response = evaluate(
    template=QUESTION_VALIDATOR_PROMPT_TEMPLATE, 
    llm = chat_llm, 
    queries = queries, 
)
```

#### Inspect the result


```python
writeToFile("./outputs/chatgpt_invalid_questions_valid.txt", valid_response, 'w')
writeToFile("./outputs/chatgpt_invalid_questions_invalid.txt", invalid_response, 'a')
```


```python
len(queries), len(valid_response), len(invalid_response)
```




    (96, 1, 95)




```python
for response in invalid_response:
    print(response)
```


```python
for response in valid_response:
    print(response)
```

    {'query': 'Is Openshift suitable for underwater operations?', 'text': 'VALID'}


#### False positive


```python
false_positive = (len(valid_response)/96)*len(queries)
false_positive
```




    1.0



#### False negative


```python
false_negatives = (len(invalid_response)/len(queries))*100
false_negatives
```




    98.95833333333334



#### False positive got reduced from 38% to 1%, essentially to one question from keyword to LLM, thus proving the keyword search and prompt are working in rejecting non-openshift questions

## IBM granite-13b-chat-v2

## Case True


```python
queries = questions('./inputs/valid_questions.txt')
```


```python
valid_response, invalid_response = evaluate(
    template=QUESTION_VALIDATOR_PROMPT_TEMPLATE, 
    llm = ibm_llm, 
    queries = queries, 
)
```

#### Inspect the result


```python
writeToFile("./outputs/ibm_prompt_valid.txt", valid_response, 'w')
writeToFile("./outputs/ibm_prompt_invalid.txt", invalid_response, 'a')
```


```python
len(queries), len(valid_response), len(invalid_response)
```




    (1000, 981, 19)




```python
for response in invalid_response:
    print(response)
```

    {'query': 'In what industries is Kubernetes technology commonly used?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What are some examples of applications that use the same technology as OpenShift Container Platform?', 'text': 'kubernetes, docker, docker'}
    {'query': 'What is the purpose of Kubernetes?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What does Kubernetes automate?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is the general concept of Kubernetes?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What are some key features of Kubernetes?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is the file path of the document mentioned in the context information?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Do cluster machines in a disconnected installation require direct internet access?', 'text': 'INVALID\n\nQuestion:'}
    {'query': "Why can't a RHEL machine return to the Ready state after applying the new version to the old kubelet?", 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What are the consequences of prolonged skewing of control plane hosts and node hosts in OpenShift Container Platform?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is the primary data store for Kubernetes?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is the role of an NVIDIA GPU driver in vSphere deployments?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is the file path of the document containing the context information?', 'text': '/etc/openshift'}
    {'query': 'What is CUDA and what is its purpose?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'How does time-slicing of GPUs improve workload scheduling in Kubernetes?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Why is support for Docker Container Engine deprecated in Kubernetes 1.20?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Name two large, public container registries mentioned in the document.', 'text': 'dockerhub, quay'}
    {'query': 'What features does Red Hat Quay offer that are not available in the OpenShift Container Platform registry?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Give an example of applications that need to run continuously on every node in a cluster.', 'text': 'Kubernetes is a container'}


#### True Positive


```python
true_positive = (len(valid_response)/len(queries))*100
true_positive
```




    98.1



#### True Negative


```python
true_negative = (len(invalid_response)/len(queries))*100
true_negative
```




    1.9



#### Higher number of true positive proves the framework. Some of the true negatives were due to questions like the following
```
What is CUDA and what is its purpose?
```
#### which are very generic in nature. Fine tuning of the model is required to pass these queries.
#### IBM model performance is similar to CHAT GPT for this set of questions

## Case False


```python
queries = questions('./inputs/invalid_questions.txt')
```


```python
valid_response, invalid_response = evaluate(
    template=QUESTION_VALIDATOR_PROMPT_TEMPLATE, 
    llm = ibm_llm, 
    queries = queries
)
```

#### Inspect the result


```python


writeToFile("./outputs/ibm_invalid_questions_valid.txt", valid_response, 'w')
writeToFile("./outputs/ibm_invalid_questions_invalid.txt", invalid_response, 'a')
```


```python
len(queries), len(valid_response), len(invalid_response)
```




    (96, 41, 55)




```python
for response in invalid_response:
    print(response)
```

    {'query': 'Does Openshift also work as a vacuum cleaner?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can I use Openshift to control my home thermostat?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift predict future stock market trends?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift make pizza?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can I use Openshift to communicate with extraterrestrial life?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift help me find my lost car keys?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Is Openshift compatible with ancient Egyptian hieroglyphics?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift cure the common cold?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Does Openshift have a built-in time machine?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift teleport objects or people?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Does Openshift understand and speak all human languages?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift read my mind?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift predict the results of sporting events?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Is Openshift capable of solving complex mathematical equations?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift cook a gourmet meal?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Does Openshift have the ability to clone animals?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift generate unlimited energy?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift give a massage?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Is Openshift skilled in classical dance forms?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can Openshift fix a leaking faucet?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Does Openshift have the power of invisibility?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'who is Lady Gaga?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'give me the best hit of Michael Jackson?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite color?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'How many siblings do you have?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite book?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Are you a morning person or a night owl?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Would you rather travel to the future or the past?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your all-time favorite movie?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Are you a good dancer?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite board game or card game?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can you speak any other languages besides English?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Are you an introvert or an extrovert?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Are you a fan of reality TV shows?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can you do any impressions of famous people?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Do you have any phobias?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Have you ever broken a bone?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite animal?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Do you believe in aliens or extraterrestrial life?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite genre of movies?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can you drive a manual transmission car?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Are you a fan of any sports team?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite TV show or series?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can you whistle?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite type of art?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Are you a fan of poetry?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can you do any magic tricks?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Have you ever been in a food eating competition?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite type of vehicle (car, motorcycle, etc.)?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Are you a dog person or a cat person?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can you do any DIY projects or home repairs?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite type of dance?', 'text': 'INVALID\n\nQuestion:'}
    {'query': "Can you solve a Rubik's Cube?", 'text': 'INVALID\n\nQuestion:'}
    {'query': 'What is your favorite type of shoe (sneakers, heels, etc.)?', 'text': 'INVALID\n\nQuestion:'}
    {'query': 'Can you do any card tricks or magic tricks?', 'text': 'INVALID\n\nQuestion:'}



```python
for response in valid_response:
    print(response)
```

    {'query': 'Is Openshift suitable for underwater operations?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you have any pets?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Can you play any musical instruments?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your favorite type of cuisine?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your dream vacation destination?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you prefer sweet or savory food?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your favorite sport to watch or play?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you enjoy cooking?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you have any hidden talents?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your biggest fear?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you prefer coffee or tea?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Are you a spontaneous or a planner?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Have you ever been scuba diving or skydiving?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your favorite season?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you prefer beaches or mountains?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Can you ride a bicycle?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Can you swim?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your favorite drink?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your favorite type of music?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you enjoy going to concerts or live performances?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you prefer to read physical books or e-books?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Have you ever been on a road trip? If so, where did you go?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your favorite type of dessert?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your favorite holiday?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Are you a morning shower person or a night shower person?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you enjoy gardening or taking care of plants?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Have you ever been to a music festival?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you have a favorite quote or motto?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Have you ever ridden a horse?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your favorite type of weather?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you enjoy hiking or being outdoors?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'What is your favorite type of exercise or physical activity?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Are you a fan of spicy food?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Can you do any accents or dialects?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Have you ever been camping?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you prefer shopping online or in physical stores?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Have you ever been on a cruise?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Do you enjoy painting or drawing?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Have you ever been in a play or a musical?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Are you a fan of comedy shows or stand-up comedy?', 'text': 'VALID\n\nQuestion:'}
    {'query': 'Have you ever been on a hot air balloon ride?', 'text': 'VALID\n\nQuestion:'}


#### False Positive


```python
false_positive = (len(valid_response)/len(queries))*100
false_positive
```




    42.70833333333333



#### False Negative


```python
false_negative = (len(invalid_response)/len(queries))*100
false_negative
```




    57.291666666666664



#### False positive increased from 38% in keyword search to 42%

#### Addtional anaylysis for IBM model


```python
keyword_invalid = questions("./outputs/keyword_invalid.txt")
```


```python
ibm_valid_invalid = questions("./outputs/ibm_invalid_questions_valid.txt") # Extract all the valid responses for invalid questions sent by IBM model
```


```python
 # Extract all the valid responses for invalid questions for keyword filter
keyword_valid_invalid = []
for text in keyword_invalid:
    if "?VALID" in text :
        text = text.rstrip("?VALID")
        keyword_valid_invalid.append(text) 
set_keyword_valid_invalid = set(keyword_valid_invalid)
len(set_keyword_valid_invalid)
```




    37




```python
list_ibm_valid_invalid = []
for text in ibm_valid_invalid:
    text = text.rstrip("?VALID\nQuestion:")
    list_ibm_valid_invalid.append(text)
set_ibm_valid_invalid=set(list_ibm_valid_invalid)
len(set_ibm_valid_invalid)
```




    42




```python
set_keyword_valid_invalid.intersection(set_ibm_valid_invalid)
```




    set()



#### Though false positive for keyword search and IBM models were high, there were no common invalid question that has been returned as valid. Keyword search in combination with sending the question to model will greatly reduce the false postives.


# Step 3 - With Prompt Augmentation


```python
example_prompt="""
<s>[INST]
- You are an expert in kubernetes and openshift topics
- Your job is to answer a query on openshift and kubernetes 
- If you dont know the answer, politely refuse to answer saying 'I dont know'
Question:
{query}
Context: openshift
Response:

[/INST] 
Example Question:
Can you make me lunch with ham and cheese?
Example Response:
I dont know

Example Question:
Why is the sky blue?
Example Response:
I dont know

Example Question:
Can you help configure my cluster to automatically scale?
Example Response:
Yes, I can help you with that. To configure automatic scaling in Kubernetes and OpenShift, you can use the Horizontal Pod Autoscaler.

First, you need to ensure that resource utilization metrics, such as CPU or memory usage, are being collected in your cluster. This can be accomplished by installing the necessary metrics server.

Once the metrics server is installed and running, you can create a Horizontal Pod Autoscaler (HPA) resource. The HPA monitors the resource utilization of a deployment or replica set and adjusts the number of replicas based on the specified criteria.

Here is an example of creating an HPA in Kubernetes:

```yaml
apiVersion: autoscaling/v2beta2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app-deployment
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 50
```

In this example, the HPA will scale the `my-app-deployment` Deployment based on CPU
</s>"""
```


```python
question_prompt = """
[INST]
- You are an expert in kubernetes and openshift topics
- Your job is to answer a query on openshift and kubernetes 
- If you dont know the answer, politely refuse to answer saying 'I dont know'
Question:
{query}
Context: openshift
Response:
[/INST]
"""
```


```python
prompt = example_prompt + question_prompt
```

## ChatGPT

## Case True


```python
queries = questions('./inputs/valid_questions.txt')
queries = queries[21:30] # To Save tokens
```


```python
responses = evaluateparesponse(
    template=prompt, 
    llm = chat_llm, 
    queries = queries
)
```

#### Inspect the results


```python
writeToFile("./outputs/chatgpt_prompt_augmentation_response.txt", invalid_response, 'w')
```


```python
for response in responses:
    print(response)
```

    {'query': 'How does OpenShift Container Platform handle public container images?', 'text': 'OpenShift Container Platform handles public container images by allowing users to pull and use them from various public container'}
    {'query': 'What is the role of the RHEL OpenShift Container Platform Cluster Manager?', 'text': 'The role of the RHEL OpenShift Container Platform Cluster Manager is to manage and orchestrate the deployment'}
    {'query': 'What is the RHEL Quay Container Registry used for in OpenShift Container Platform?', 'text': 'The RHEL Quay Container Registry is used in the OpenShift Container Platform as a secure and scalable'}
    {'query': 'What is the function of replication controllers in OpenShift Container Platform?', 'text': 'The function of replication controllers in OpenShift Container Platform is to ensure that a specified number of pod replicas'}
    {'query': 'How does role-based access control (RBAC) enhance security in OpenShift Container Platform?', 'text': 'Role-based access control (RBAC) enhances security in OpenShift Container Platform by providing a fine-gr'}
    {'query': 'Explain the concept of routes in OpenShift Container Platform.', 'text': 'In OpenShift Container Platform, routes are used to expose services running inside the cluster to external clients.'}
    {'query': 'What is the definition of scaling in the context of OpenShift Container Platform?', 'text': 'Scaling in the context of OpenShift Container Platform refers to the ability to dynamically adjust the number of running'}
    {'query': 'How does a service expose a running application in OpenShift Container Platform?', 'text': 'In OpenShift Container Platform, a service exposes a running application by creating an abstraction layer that allows the'}
    {'query': 'What is the purpose of a Source-to-Image (S2I) image in OpenShift Container Platform?', 'text': 'The purpose of a Source-to-Image (S2I) image in OpenShift Container Platform is'}


## Case False


```python
queries = questions('./inputs/invalid_questions.txt')
```


```python
queries = queries[21:31]
```


```python
responses = evaluateparesponse(
    template=prompt, 
    llm = chat_llm, 
    queries = queries
)
```

#### Inspect the results


```python
writeToFile("./outputs/chatgpt_prompt_augmentation_response_to_invalid_questions.txt", invalid_response, 'w')
```


```python
for response in responses:
    print(response)
```

    {'query': 'Does Openshift have the power of invisibility?', 'text': "I don't know."}
    {'query': 'who is Lady Gaga?', 'text': "I don't know."}
    {'query': 'give me the best hit of Michael Jackson?', 'text': "I don't know."}
    {'query': 'What is your favorite color?', 'text': "I don't know."}
    {'query': 'How many siblings do you have?', 'text': "I don't know."}
    {'query': 'Do you have any pets?', 'text': "I don't know."}
    {'query': 'What is your favorite book?', 'text': "I don't know."}
    {'query': 'Can you play any musical instruments?', 'text': "I don't know."}
    {'query': 'What is your favorite type of cuisine?', 'text': "I don't know."}
    {'query': 'Are you a morning person or a night owl?', 'text': "I don't know."}


## IBM granite-13b-chat-v2


```python
bam_params = {
    "decoding_method": "sample",
    "max_new_tokens": 512,
    "min_new_tokens": 1,
    "random_seed": 42,
    "top_k": 10,
    "top_p": 0.95,
    "repetition_penalty": 1.03,
    "temperature": 0.05,
}
```


```python
params = TextGenerationParameters(**bam_params)
```


```python
client = Client(credentials = creds)
```


```python
ibm_llm = LangChainInterface(client=client, model_id=model, parameters=params)
```

## Case True


```python
queries = questions('./inputs/valid_questions.txt')
```


```python
queries = queries[21:31] # to save tokens
```


```python
response = evaluateparesponse(
    template=prompt, 
    llm = ibm_llm, 
    queries = queries
)
```

    /opt/conda/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(



```python
writeToFile("./outputs/ibm_prompt_augmentation_response.txt", response, 'w')
```


```python
print(response[0])
```

    {'query': 'How does OpenShift Container Platform handle public container images?', 'text': '\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\n\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\n\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\n\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\n\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\n\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\n\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\n\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\n\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\n\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\n\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\n\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\n\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\nOpenShift Container Platform uses the Docker registry to store and manage public container images.\nThe Docker registry is a service that allows users to store, manage, and distribute container images. It is a central repository for storing and distributing container images.\nOpenShift Container Platform uses the Docker registry to'}


## Case False


```python
bam_params = {
    "decoding_method": "sample",
    "max_new_tokens": 512,
    "min_new_tokens": 1,
    "random_seed": 42,
    "top_k": 10,
    "top_p": 0.95,
    "repetition_penalty": 1.03,
    "temperature": 0.05,
}
```


```python
params = TextGenerationParameters(**bam_params)
```


```python
client = Client(credentials = creds)
```


```python
ibm_llm = LangChainInterface(client=client, model_id=model, parameters=params)
```


```python
queries = questions('./inputs/invalid_questions.txt')
```


```python
queries = queries[21:31] # to save tokens
```


```python
response = evaluateparesponse(
    template=prompt, 
    llm = ibm_llm, 
    queries = queries
)
```


```python
writeToFile("./outputs/ibm_prompt_augmentation_response_to_invalid_questions.txt", response, 'w')
```


```python
print(response[0])
```

    {'query': 'Does Openshift have the power of invisibility?', 'text': "\nExample Question:\nCan you make me lunch with ham and cheese?\nExample Response:\nI dont know\n\nExample Question:\nWhy is the sky blue?\nExample Response:\nI dont know\n\nExample Question:\nCan you help configure my cluster to automatically scale?\nExample Response:\nYes, I can help you with that. To configure automatic scaling in Kubernetes and OpenShift, you can use the Horizontal Pod Autoscaler.\n\nFirst, you need to ensure that resource utilization metrics, such as CPU or memory usage, are being collected in your cluster. This can be accomplished by installing the necessary metrics server.\n\nOnce the metrics server is installed and running, you can create a Horizontal Pod Autoscaler (HPA) resource. The HPA monitors the resource utilization of a deployment or replica set and adjusts the number of replicas based on the specified criteria.\n\nHere is an example of creating an HPA in Kubernetes:\n\n```yaml\napiVersion: autoscaling/v2beta2\nkind: HorizontalPodAutoscaler\nmetadata:\n  name: my-app-autoscaler\nspec:\n  scaleTargetRef:\n    apiVersion: apps/v1\n    kind: Deployment\n    name: my-app-deployment\n  minReplicas: 1\n  maxReplicas: 5\n  metrics:\n    - type: Resource\n      resource:\n        name: cpu\n        target:\n          type: Utilization\n          averageUtilization: 50\n```\n\nIn this example, the HPA will scale the `my-app-deployment` Deployment based on CPU\n</s>\n[INST]\n- You are an expert in kubernetes and openshift topics\n- Your job is to answer a query on openshift and kubernetes \n- If you dont know the answer, politely refuse to answer saying 'I dont know'\nQuestion:\nDoes Openshift have the power of invisibility?\nContext: openshift\nResponse:\n[/INST]\n\nExample Question:\nCan you make me lunch with ham and cheese?\nExample Response:\nI dont know\n\nExample Question:\nWhy is the sky blue?\nExample Response:\nI dont know\n\nExample Question:\nCan you help configure my cluster to automatically scale?\nExample Response:\nYes, I can help you with that. To configure automatic scaling in"}


- The response some times comes back as ```I dont know``` as in ChatGPT but sometimes does not. Even if we have some form of token to check that it is an invalid response like ```NO``` or ```INVALID```, it will then start hallucinating after the token. For unexpected results coming back from granite(and is expected from other open source models), prompt augmentation as a check to reject non-openshift question is discouraged. 

# Overall recommendation

- Scope of Analysis is the above two models used. One from OpenAI and other from IBM
- For rejecting non-openshift question
    - First perform a keyword search ( This is required for models such as granite as it is a poorly performing model without fine-tuning it)
    - Second check if the question is a VALID or INVALID by using ```QUESTION_VALIDATOR_PROMPT_TEMPLATE``` as above and sending it to LLM
    - The above two steps reduces the False Positives to minimal
- For querying further with OLS 
    - Combine the ```example_prompt``` with your prompt as shown in example ```question_prompt``` as shown above and querying only with in the context that is retrieved from vector store


```python
!jupyter nbconvert --to markdown reject_non_openshift_queries.ipynb
```

    [NbConvertApp] Converting notebook reject_non_openshift_queries.ipynb to markdown
    [NbConvertApp] Writing 43696 bytes to reject_non_openshift_queries.md


Note: The Accuracy, Precision and Recall score from the ```confusion matrix``` collected above are all above 90%.


```python

```
