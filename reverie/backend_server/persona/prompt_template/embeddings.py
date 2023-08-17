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
    def __init__(self, model, exception_handler = None):
      self.model = model
      self.exception_handler = exception_handler
    def embed_query(self, text):
      log.debug(f'{text=}')
      try:
        return self.model.embed_query(text)
      except Exception as e:
        if self.exception_handler:
          self.exception_handler(e)
        else:
          raise e
except:
  pass
