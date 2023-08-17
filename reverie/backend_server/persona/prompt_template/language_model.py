"""
Author: irrealis (irrealis.chomp@gmail.com)

File: language_model.py
Description: Defines simple base class to wrap LLM calls.
"""

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class LanguageModel:
  def generate(str):
    raise NotImplementedError


#try:
#  import openai
#
#  class OpenAIChatModel(LanguageModel):
#    def __init__(self, model_name, **kw):
#      self.model_name = model_name
#    def generate(self, prompt):
#      completion = openai.ChatCompletion.create(
#        model=self.model_name, 
#        messages=[{"role": "user", "content": prompt}],
#        **kw
#      )
#      return completion["choices"][0]["message"]["content"]
#
#  class OpenAIModel(LanguageModel):
#    def __init__(self, model_name, **kw):
#      self.model_name = model_name
#    def generate(self, prompt):
#      completion = openai.Completion.create(
#        model=self.model_name, 
#        messages=[{"role": "user", "content": prompt}],
#        **kw
#      )
#      return completion["choices"][0]["message"]["content"]
#except:
#  pass
#
#
#try:
#  import gpt4all
#
#  class GPT4AllModel(LanguageModel):
#    def __init__(self, model_name):
#      self.model = gpt4all.GPT4All(model_name)
#    def generate(self, prompt, **kw):
#      output = model.generate(prompt, **kw)
#      return output
#except:
#  pass


try:
  import langchain

  class LangChainModel(LanguageModel):
    def __init__(self, model, exception_handler = None):
      self.model = model
      self.exception_handler = exception_handler
    def generate(self, prompt, **kw):
      if isinstance(self.model, langchain.chat_models.base.BaseChatModel):
        try:
          output = self.model(
            [
              langchain.schema.HumanMessage(content=prompt, additional_kwargs=kw),
            ],
          )
        except Exception as e:
          if self.exception_handler:
            self.exception_handler(e)
          else:
            raise e
      elif isinstance(self.model, langchain.llms.base.BaseLLM):
        try:
          output = self.model(prompt, **kw)
        except Exception as e:
          if self.exception_handler:
            self.exception_handler(e)
          else:
            raise e
      else:
        raise Exception('model is neither chat nor llm')
      return output.content
except:
  pass
