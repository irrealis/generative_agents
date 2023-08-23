import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from ..ablate import (
  ablate_observations_planning_reflection,
  ablate_planning_reflection,
  ablate_reflection
)
from .believability_questions import get_believability_question_variables
from ..interview import interview_persona

import jsonpickle

import ruamel
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

import json, os


yaml = YAML()

this_dir = os.path.abspath(os.path.dirname(__file__))


class Conditions(object):
  def __init__(self, persona):
    persona_frozen = jsonpickle.encode(persona, indent=2)
    self.no_observation_no_reflection_no_planning = jsonpickle.decode(persona_frozen)
    self.no_reflection_no_planning = jsonpickle.decode(persona_frozen)
    self.no_reflection = jsonpickle.decode(persona_frozen)
    self.full_architecture = jsonpickle.decode(persona_frozen)

    ablate_observations_planning_reflection(self.no_observation_no_reflection_no_planning)
    ablate_planning_reflection(self.no_reflection_no_planning)
    ablate_reflection(self.no_reflection)

  def interview__no_observation_no_reflection_no_planning(self, question, **kw):
    return interview_persona(self.no_observation_no_reflection_no_planning, question, **kw)

  def interview__no_reflection_no_planning(self, question, **kw):
    return interview_persona(self.no_reflection_no_planning, question, **kw)

  def interview__no_reflection(self, question, **kw):
    return interview_persona(self.no_reflection, question, **kw)

  def interview__full_architecture(self, question, **kw):
    return interview_persona(self.full_architecture, question, **kw)

  def get_condition_methods_dict(self):
    condition_methods_dict = dict(
      no_observation_no_reflection_no_planning = self.interview__no_observation_no_reflection_no_planning,
      no_reflection_no_planning = self.interview__no_reflection_no_planning,
      no_reflection = self.interview__no_reflection,
      full_architecture = self.interview__full_architecture,
    )
    return condition_methods_dict


class BelievabilityInterviewer(object):
  # Layout of the interviews dict:
  #
  # interviews = {
  #   'personas': [
  #     {
  #       'persona': <persona.name>
  #       'categories': [
  #         {
  #           'category': <category>,
  #           'questions': [
  #             {
  #               'question': <question>
  #               'conditions': [
  #                 'condition': <condition>
  #                 'response': <response>
  #               ]
  #             }
  #           ]
  #         }
  #       ]
  #     }
  #   ]
  # }
  #
  def __init__(
    self,
    question_templates,
    personas,
    random_persona_clause,
    event,
    random_seed
  ):
    self.question_templates = question_templates
    self.personas = personas
    self.random_persona_clause = random_persona_clause
    self.event = event
    self.random_seed = random_seed

  def generate_condition_dict(
    self,
    question,
    condition,
    method,
  ):
    retrieved, summarized_idea, response, curr_conv = method(question)
    condition_dict = dict(
      condition = condition,
      response = response,
      summarized_idea = summarized_idea,
    )
    return condition_dict

  def generate_question_dict(
    self,
    question_id,
    template,
    persona,
    condition_methods,
  ):
    question_variables = get_believability_question_variables(
      persona=persona,
      personas=self.personas,
      random_persona_clause=self.random_persona_clause,
      event=self.event,
      random_seed=self.random_seed,
    )
    question = template.format_map(question_variables)
    question_dict = dict(
      question_id = question_id,
      question = question,
      conditions = list()
    )
    for condition, method in condition_methods.items():
      # This calls one of the Condition methods, which in turn calls the language model.
      condition_dict = self.generate_condition_dict(question, condition, method)
      question_dict['conditions'].append(condition_dict)
    return question_dict

  def generate_category_dict(
    self,
    category,
    questions,
    persona,
    condition_methods,
  ):
    category_dict = dict(
      category = category,
      questions = list()
    )
    for question_id, template in questions.items():
      question_dict = self.generate_question_dict(question_id, template, persona, condition_methods)
      category_dict['questions'].append(question_dict)
    return category_dict

  def generate_persona_dict(self, persona):
    conditions = Conditions(persona)
    condition_methods = conditions.get_condition_methods_dict()
    persona_dict = dict(
      persona = persona.name,
      categories = list()
    )
    for category, questions in self.question_templates.items():
      category_dict = self.generate_category_dict(category, questions, persona, condition_methods)
      persona_dict['categories'].append(category_dict)
    return persona_dict

  def generate_interviews_dict(self, personas = None):
    if personas is None:
      personas = self.personas
    interviews_dict = dict(
      interviews = dict(
        personas = list()
      )
    )
    for persona in personas.values():
      persona_dict = self.generate_persona_dict(persona)
      interviews_dict['interviews']['personas'].append(persona_dict)
    return interviews_dict



def believability_interviews(personas, sim_folder, random_seed=None):
  interview_questions_path = os.path.join(this_dir, 'V1_interview_questions', 'believability_templates.json')
  believability_dir = os.path.join(sim_folder, 'analysis', 'believability')
  interviews_path = os.path.join(believability_dir, 'interviews.yaml')
  persona_names = list(personas.keys())

  os.makedirs(believability_dir, exist_ok=True)
  with open(interview_questions_path, 'rb') as f:
    question_templates = json.load(f)

  interviewer = BelievabilityInterviewer(
    question_templates=question_templates,
    personas=personas,
    random_persona_clause="organizing a Valentine's Day party",
    event="a Valentine's Day party",
    random_seed=random_seed,
  )
  interviews_dict = interviewer.generate_interviews_dict()

  with open(interviews_path, 'w') as f:
    yaml.dump(interviews_dict, f)
