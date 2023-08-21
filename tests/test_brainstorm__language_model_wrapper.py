import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from dotenv import find_dotenv, load_dotenv
env_path = find_dotenv()
load_dotenv(env_path)

import os
import sys

project_dir = os.path.dirname(os.path.abspath(env_path))
sys.path.insert(0, os.path.abspath(f"{project_dir}/reverie/backend_server"))

from langchain_setup import *

from reverie import ReverieServer
from persona.persona import Persona
from persona.prompt_template.language_model import LangChainModel
from persona.prompt_template.embeddings import LangChainEmbeddings

import langchain
from langchain.cache import SQLiteCache, RETURN_VAL_TYPE
from langchain.chains.base import Chain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from pydantic import BaseModel, Field
import pytest

import random


@pytest.fixture
def oai_chat_model():
  return ChatOpenAI()

@pytest.fixture
def oai_embeddings():
  return OpenAIEmbeddings()

@pytest.fixture
def hf_embeddings():
  return HuggingFaceEmbeddings(
    model_name="multi-qa-distilbert-cos-v1",
    model_kwargs={'device':'cuda:0'}
  )


@pytest.fixture
def lm_oai_chat_model(oai_chat_model):
  return LangChainModel(oai_chat_model)

@pytest.fixture
def embs_oai(oai_embeddings):
  return LangChainEmbeddings(oai_embeddings)


random.seed(0)


def test_brainstorm__embeddings_wrapper__embs_oai(embs_oai):
  result = embs_oai.embed_query('hi.')
  log.debug(f'{len(result)=}')


def test_brainstorm__language_model_wrapper__lm_oai_chat_model(lm_oai_chat_model):
  result = lm_oai_chat_model.generate('hi.')
  log.debug(f'{result=}')


def test_brainstorm__language_model_wrapper__oai_chat_model(oai_chat_model):
  lcm = LangChainModel(oai_chat_model)
  result = lcm.generate('hi.')
  log.debug(f'{result=}')

