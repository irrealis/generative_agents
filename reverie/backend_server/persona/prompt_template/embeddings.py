"""
Author: irrealis (irrealis.chomp@gmail.com)

File: embeddings.py
Description: Defines simple base class to wrap embedding calls.
"""

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Embeddings:
  def embed_query(text):
    raise NotImplementedError


try:
  import langchain

  class LangChainEmbeddings(Embeddings):
    def __init__(self, model):
      self.model = model
    def embed_query(self, text):
      log.debug(f'{text=}')
      return self.model.embed_query(text)
except:
  pass
