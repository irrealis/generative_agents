import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from dotenv import find_dotenv, load_dotenv
env_path = find_dotenv()
load_dotenv(env_path)

import os
import sys

project_dir = os.path.dirname(env_path)
log.debug(f'{project_dir=}')

backend_server_loc = os.path.abspath(f"{project_dir}/reverie/backend_server")
sys.path.insert(0, backend_server_loc)

irrealis_datascience_loc = os.path.abspath(f"{project_dir}/../datascience")
sys.path.insert(0, irrealis_datascience_loc)


from reverie import ReverieServer
from persona.persona import Persona
from persona.prompt_template.language_model import LangChainModel
from persona.prompt_template.embeddings import LangChainEmbeddings

#from irrealis.retrievers import TimeWeightedVectorStoreRetriever
#from irrealis.vectorstores import PGVector
#from irrealis.generative_agents.agent import *
#from irrealis.generative_agents.chains import *
#from irrealis.generative_agents.gender import *
#from irrealis.generative_agents.prompts import *
from irrealis.generative_agents.test_tools import *

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

import datetime as dt
import re


langchain.llm_cache = SQLiteCache_ForTests(database_path=".langchain.db", raise_on_miss=True)
#langchain.llm_cache = SQLiteCache_ForTests(database_path=".langchain.db", raise_on_miss=False)
#langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']='Park Generative Agents'
tags = ['test', 'brainstorm']
db_url = "postgresql+psycopg2://kaben:{DB_PASS}@localhost:5432/datascience".format(
  DB_PASS=os.getenv('DB_PASS')
)


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

