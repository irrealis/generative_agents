import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from dotenv import find_dotenv, load_dotenv
env_path = find_dotenv()
load_dotenv(env_path)

from typing import Optional
import os


project_dir = os.path.dirname(os.path.abspath(env_path))


# Try to import LangChain. If this works, setup cache for LLM calls.
try:
  import langchain
  from langchain.cache import SQLiteCache, RETURN_VAL_TYPE

  log.debug('Successfully imported LangChain.')


  class LangChainCacheMissException(Exception):
    def __init__(self, message):
      super().__init__(message)


  class SQLiteCache_ForTests(SQLiteCache):
    '''
    Class that verifies cache retrieval during tests.

    Raises assert exception when cache lookup fails.

    Problem: This adds an element of nondeterminism to the tests below, but these
    tests generate enough LLM calls to be costly when there are a lot of cache
    misses. So I want to know when this happens.

    Note: if a lookup fails, it will cause the test to fail, but upon a second
    test run, the lookup should succeed if the source code hasn't changed. This
    means:

    - I'll only receive one notification when a cache miss has occurred.
    - Test code that intentionally sends a new LLM prompt will fail once, but
    subsequently will pass.
    '''

    def __init__(self, *l, raise_on_miss=False, **d):
      super().__init__(*l, **d)
      self.raise_on_miss = raise_on_miss
      self.hit_miss_tracker = None

    def track_hit(self, prompt, llm_str):
      if isinstance(self.hit_miss_tracker, list):
        self.hit_miss_tracker.append(dict(
          hit_miss = 'HIT',
          prompt = prompt,
          llm_str = llm_str,
        ))

    def track_miss(self, prompt, llm_str):
      if isinstance(self.hit_miss_tracker, list):
        self.hit_miss_tracker.append(dict(
          hit_miss = 'MISS',
          prompt = prompt,
          llm_str = llm_str,
        ))

    def lookup(self, prompt: str, llm_str: str) -> Optional[RETURN_VAL_TYPE]:
      result = super().lookup(prompt=prompt, llm_string=llm_str)
      if result:
        self.track_hit(prompt, llm_str)
      else:
        self.track_miss(prompt, llm_str)
        if self.raise_on_miss:
          raise LangChainCacheMissException(
            message=f'''
LLM cache MISS:
Prompt:
```
{prompt=}
```
LLM string:
```
{llm_str=}
```
'''.strip()
          )
        else:
          log.warning(
          f'''
LLM cache MISS:
Prompt:
```
{prompt=}
```
LLM string:
```
{llm_str=}
```
'''
          )
      return result

  langchain.llm_cache = SQLiteCache_ForTests(database_path=f'{project_dir}/.langchain.db', raise_on_miss=False)
  os.environ['LANGCHAIN_TRACING_V2']='true'
  os.environ['LANGCHAIN_PROJECT']='Park Generative Agents - Tests'


except ModuleNotFoundError:
  log.info(
    '''
Unable to import setup LangChain tools; LangChain not installed.
Not setting up LangChain caching for LLM calls.
'''
  )
