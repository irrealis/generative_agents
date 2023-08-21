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


# Try to import LangChain. If this works, verify that LLM caching is setup.
try:
  import langchain
  def test_verify_langchain_cache():
    log.debug(f'{langchain.llm_cache=}')
    assert isinstance(langchain.llm_cache, SQLiteCache_ForTests)
except ModuleNotFoundError:
  pass
