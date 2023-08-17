"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import json
import random
import openai
import openai.error
import time 

from utils import *
from .language_model import LangChainModel
from .embeddings import LangChainEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


def openai_exception_handler(exception):
  log.error(f'Error; exception: {exception}')
  raise exception

openai.api_key = openai_api_key
llm_oai_gpt_35_turbo = LangChainModel(ChatOpenAI(), openai_exception_handler)
llm_oai_gpt_4 = LangChainModel(ChatOpenAI(model_name='gpt-4'), openai_exception_handler)
embeddings_oai_ada = LangChainEmbeddings(OpenAIEmbeddings(model='text-embedding-ada-002'), openai_exception_handler)



def temp_sleep(seconds=0.1):
  time.sleep(seconds)

# NOTE@kaben: Uses:
# - plan.py
#   - 4 calls
def ChatGPT_single_request(prompt): 
  temp_sleep()

  #completion = openai.ChatCompletion.create(
  #  model="gpt-3.5-turbo", 
  #  messages=[{"role": "user", "content": prompt}]
  #)
  #return completion["choices"][0]["message"]["content"]
  return llm_oai_gpt_35_turbo.generate(prompt)


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

# NOTE@kaben: Only referenced in this file.
def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  #try: 
  #  completion = openai.ChatCompletion.create(
  #  model="gpt-4", 
  #  messages=[{"role": "user", "content": prompt}]
  #  )
  #  return completion["choices"][0]["message"]["content"]
  #
  #except: 
  #  print ("ChatGPT ERROR")
  #  return "ChatGPT ERROR"
  return llm_oai_gpt_4.generate(prompt)


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  #try: 
  #  completion = openai.ChatCompletion.create(
  #  model="gpt-3.5-turbo", 
  #  messages=[{"role": "user", "content": prompt}]
  #  )
  #  return completion["choices"][0]["message"]["content"]
  #
  #except: 
  #  print ("ChatGPT ERROR")
  #  return "ChatGPT ERROR"
  result = llm_oai_gpt_35_turbo.generate(prompt)
  return result


# NOTE@kaben: Not used.
def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


# NOTE@kaben: Uses:
# - run_gpt_prompt.py
#   - 12 calls
def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except openai.error.InvalidRequestError as e:
      log.error(f'OpenAI InvalidRequestError; exception: {e}')
      raise e
    except Exception as e: 
      pass

  return False


# NOTE@kaben: Uses:
# - run_gpt_prompt.py
#   - 2 calls
def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

# NOTE@kaben: Uses:
# - this file
#   - 1 calls
def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try: 
    #response = openai.Completion.create(
    #            model=gpt_parameter["engine"],
    #            prompt=prompt,
    #            temperature=gpt_parameter["temperature"],
    #            max_tokens=gpt_parameter["max_tokens"],
    #            top_p=gpt_parameter["top_p"],
    #            frequency_penalty=gpt_parameter["frequency_penalty"],
    #            presence_penalty=gpt_parameter["presence_penalty"],
    #            stream=gpt_parameter["stream"],
    #            stop=gpt_parameter["stop"],)
    #return response.choices[0].text
    llm = LangChainModel(ChatOpenAI(
      model_name=gpt_parameter["engine"],
      temperature=gpt_parameter["temperature"],
      max_tokens=gpt_parameter["max_tokens"],
      streaming=gpt_parameter["stream"],
      model_kwargs=dict(
        top_p=gpt_parameter["top_p"],
        frequency_penalty=gpt_parameter["frequency_penalty"],
        presence_penalty=gpt_parameter["presence_penalty"],
        stop=gpt_parameter["stop"],
      ),
    ))
    return llm.generate(prompt)
  except: 
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


# NOTE@kaben: Uses:
# - gpt_structure.py
#   - 2 calls
# - run_gpt_prompt.py
#   - 36 calls
def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    # TODO@kaben: Refactor.
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  #return openai.Embedding.create(
  #        input=[text], model=model)['data'][0]['embedding']
  return LangChainEmbeddings(OpenAIEmbeddings(model=model)).embed_query(text)


if __name__ == '__main__':
  gpt_parameter = {"engine": "gpt-3.5-turbo", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















