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


def generate_condition_dict(
  question,
  condition,
  method,
):
  response, curr_conv = method(question)
  condition_dict = dict(
    condition = condition,
    response = response,
  )
  return condition_dict


def generate_question_dict(
  question_id,
  template,
  persona,
  reverie_server,
  condition_methods,
):
  question_variables = get_believability_question_variables(
    persona=persona,
    personas=reverie_server.personas,
    random_persona_clause = "organizing a Valentine's Day party",
    event = "a Valentine's Day party",
    # Get a deterministic random number generator by seeding with 0.
    random_seed = 0,
  )
  question = template.format_map(question_variables)
  question_dict = dict(
    question_id = question_id,
    question = question,
    conditions = list()
  )
  for condition, method in condition_methods.items():
    # This calls one of the Condition methods, which in turn calls the language model.
    condition_dict = generate_condition_dict(question, condition, method)
    question_dict['conditions'].append(condition_dict)
  return question_dict


def generate_category_dict(
  category,
  questions,
  persona,
  reverie_server,
  condition_methods,
):
  category_dict = dict(
    category = category,
    questions = list()
  )
  for question_id, template in questions.items():
    question_dict = generate_question_dict(
      question_id=question_id,
      template=template,
      persona=persona,
      reverie_server=reverie_server,
      condition_methods=condition_methods,
    )
    category_dict['questions'].append(question_dict)
  return category_dict


def believability_interviews(reverie_server, sim_folder):
  interview_questions_path = os.path.join(this_dir, 'V1_interview_questions', 'believability_templates.json')
  believability_dir = os.path.join(sim_folder, 'analysis', 'believability')
  interviews_path = os.path.join(believability_dir, 'interviews.yaml')
  persona_names = list(reverie_server.personas.keys())

  os.makedirs(believability_dir, exist_ok=True)
  with open(interview_questions_path, 'rb') as f:
    believability_questions = json.load(f)

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

  ablation_conditions = dict(
    no_observation_no_reflection_no_planning = '',
    no_reflection_no_planning = '',
    no_reflection = '',
    full_architecture = '',
    human = '',
  )

  interviews_dict = dict(
    interviews = dict(
      personas = list()
    )
  )
  for persona in reverie_server.personas.values():
    conditions = Conditions(persona)
    condition_methods = conditions.get_condition_methods_dict()

    persona_dict = dict(
      persona = persona.name,
      categories = list()
    )
    for category, questions in believability_questions.items():
      category_dict = generate_category_dict(
        category=category,
        questions=questions,
        persona=persona,
        reverie_server=reverie_server,
        condition_methods=condition_methods,
      )
      #category_dict = dict(
      #  category = category,
      #  questions = list()
      #)
      #for question_id, template in questions.items():
      #  question_dict = generate_question_dict(
      #    question_id=question_id,
      #    template=template,
      #    persona=persona,
      #    reverie_server=reverie_server,
      #    condition_methods=condition_methods,
      #  )
      #  #question_variables = get_believability_question_variables(
      #  #  persona=persona,
      #  #  personas=reverie_server.personas,
      #  #  random_persona_clause = "organizing a Valentine's Day party",
      #  #  event = "a Valentine's Day party",
      #  #  # Get a deterministic random number generator by seeding with 0.
      #  #  random_seed = 0,
      #  #)
      #  #question = template.format_map(question_variables)
      #  #question_dict = dict(
      #  #  question_id = question_id,
      #  #  question = question,
      #  #  conditions = list()
      #  #)
      #  #for condition, method in condition_methods.items():
      #  #  # This calls one of the Condition methods, which in turn calls the language model.
      #  #  condition_dict = generate_condition_dict(question, condition, method)
      #  #  #response, curr_conv = method(question)
      #  #  #condition_dict = dict(
      #  #  #  condition = condition,
      #  #  #  response = response,
      #  #  #)
      #  #  question_dict['conditions'].append(condition_dict)
      #  category_dict['questions'].append(question_dict)
      persona_dict['categories'].append(category_dict)
    interviews_dict['interviews']['personas'].append(persona_dict)

  with open(interviews_path, 'w') as f:
    yaml.dump(interviews_dict, f)
