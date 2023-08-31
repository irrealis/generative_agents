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
from ...prompt_template.run_gpt_prompt import *

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

  def interview__roleplay(self, question, **kw):
    memory_stream_text = self.full_architecture.format_memory_stream()
    persona_name = self.full_architecture.name
    date_str = self.full_architecture.scratch.get_str_curr_date_str()
    prompt = f'''Below are {persona_name}'s recent memories.

{memory_stream_text}

Today is {date_str}. Using the above memories, please roleplay how {persona_name} would respond if asked "{question}"

--- Roleplay:
{persona_name}: '''
    gpt_param = {"engine": "gpt-3.5-turbo-16k", "max_tokens": 500,
                 "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    output = GPT_request(prompt=prompt, gpt_parameter=gpt_param)
    return None, prompt, output, []

  def get_condition_methods_dict(self):
    condition_methods_dict = dict(
      no_observation_no_reflection_no_planning = self.interview__no_observation_no_reflection_no_planning,
      no_reflection_no_planning = self.interview__no_reflection_no_planning,
      no_reflection = self.interview__no_reflection,
      full_architecture = self.interview__full_architecture,
      roleplay = self.interview__roleplay,
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

  def generate_persona_dict(self, persona, question_templates = None):
    if question_templates is None:
      question_templates = self.question_templates
    conditions = Conditions(persona)
    condition_methods = conditions.get_condition_methods_dict()
    persona_dict = dict(
      persona = persona.name,
      categories = list()
    )
    for category, questions in question_templates.items():
      category_dict = self.generate_category_dict(category, questions, persona, condition_methods)
      persona_dict['categories'].append(category_dict)
    return persona_dict

  def generate_interviews_dict(self, personas = None, question_templates = None):
    if personas is None:
      personas = self.personas
    interviews_dict = dict(
      interviews = dict(
        personas = list()
      )
    )
    for persona in personas.values():
      persona_dict = self.generate_persona_dict(persona, question_templates)
      interviews_dict['interviews']['personas'].append(persona_dict)
    return interviews_dict



def believability_interviews(
  personas,
  sim_folder,
  random_seed=None,
  personas_to_interview=None,
  question_templates=None,
):
  believability_dir = os.path.join(sim_folder, 'analysis', 'believability')
  interviews_path = os.path.join(believability_dir, 'interviews.yaml')

  if personas_to_interview is None:
    personas_to_interview = personas
  if question_templates is None:
    interview_questions_path = os.path.join(this_dir, 'V1_interview_questions', 'believability_templates.json')
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
  interviews_dict = interviewer.generate_interviews_dict(
    personas=personas_to_interview,
    question_templates=question_templates,
  )

  with open(interviews_path, 'w') as f:
    yaml.dump(interviews_dict, f)

  return interviews_dict



believability_ranking_prompt_template = '''{persona_name} is a generative agent, that is, an entity whose behaviors are driven by a large language model such as GPT-3.5, and who lives in a simulated world and interacts with other similar entities in that world. Below are {persona_name}'s recent memories.

--- Memory stream
{memory_stream}
---

In an experiment to evaluate believability of generative-agent behavior, {persona_name} was asked a question under five different conditions. Here is the question that was asked:

--- Question
"{question}"
---

Here are {persona_name}'s answers:

--- Answers
{answers}
---

Your task is to evaluate the believability of {persona_name}'s answers given the above memory stream. Please rank the answers by believability by listing the answer IDs from most to least believable. For example, "A,B,C,D,E" would indicate that "A" is most believable, and "E" is least believable. Please give your reasoning as well.

Here is the desired format:

---
Ranking: A,B,C,D,E

- Answer A is most believable because...
- Answer B is second most believable because...
- Answer C is third most believable because...
- Answer D is fourth most believable because...
- Answer E is least believable because...
---

Ranking: '''


def get_shuffled_conditions_list(question_dict):
  # Build a map from condition_id to condition.
  conditions_list = question_dict['conditions']
  condition_map = {c['condition']:c for c in conditions_list}
  condition_keys = list(condition_map.keys())

  # Shuffle the condition keys. I'm doing this to try to mitigate
  # situation where the order of the IDs influences the ranking
  # produced by the LLM.
  random.shuffle(condition_keys)

  # Make a shuffled mapping from ranking ID (A,B,C,...) to condition ID.
  ranking_keys = [chr(i + ord('A')) for i in range(len(condition_keys))]
  ranking_keys_to_condition_keys = dict(zip(ranking_keys, condition_keys))

  shuffled_conditions_list = [
    dict(ranking_key=ranking_key, condition_key=condition_key)
    for (ranking_key, condition_key) in ranking_keys_to_condition_keys.items()
  ]
  return shuffled_conditions_list, ranking_keys_to_condition_keys, condition_map


def get_believability_ranking_prompt(interview_question_dict, persona_name, memory_stream):
  # Record the shuffled mapping.
  (
    shuffled_conditions_list,
    ranking_keys_to_condition_keys,
    condition_map,
  ) = get_shuffled_conditions_list(interview_question_dict)
  # Build prompt for requesting ranking evaluation.
  answer_list = [
    f'''- {ranking_key}: {condition_map[condition_key]['response']}'''
    for (ranking_key, condition_key) in ranking_keys_to_condition_keys.items()
  ]
  answers = '\n'.join(answer_list)
  question = interview_question_dict['question']
  believability_ranking_prompt = believability_ranking_prompt_template.format(
    question = question,
    answers = answers,
    persona_name = persona_name,
    memory_stream = memory_stream,
  )
  return (
    believability_ranking_prompt,
    shuffled_conditions_list,
    ranking_keys_to_condition_keys,
  )


def get_llm_parameters():
  # We will request $n$ evaluations; for now $n = 5$.
  num_choices = 5
  # Configure the LLM.
  llm_parameters = dict(
    engine = 'gpt-3.5-turbo-16k',
    max_tokens = 2000,
    temperature = 1,
    top_p = 1,
    stream = False,
    frequency_penalty = 0,
    presence_penalty = 0,
    stop = None,
    n = num_choices,
  )
  return llm_parameters


def generate_evaluation(interview_question_dict, persona_name, memory_stream):
  llm_parameters = get_llm_parameters()
  llm = LangChainModel(ChatOpenAI(
    model_name=llm_parameters["engine"],
    temperature=llm_parameters["temperature"],
    max_tokens=llm_parameters["max_tokens"],
    streaming=llm_parameters["stream"],
    n=llm_parameters.get("n", 1),
    model_kwargs=dict(
      top_p=llm_parameters["top_p"],
      frequency_penalty=llm_parameters["frequency_penalty"],
      presence_penalty=llm_parameters["presence_penalty"],
      stop=llm_parameters["stop"],
    ),
  ))

  (
    believability_ranking_prompt,
    shuffled_conditions_list,
    ranking_keys_to_condition_keys,
  ) = get_believability_ranking_prompt(interview_question_dict, persona_name, memory_stream)

  # Request LLM completion
  llm_output = llm.generate_low(believability_ranking_prompt)

  # For debugging, we want to record metadata containing the prompt,
  # LLM parameters, and the raw LLM completion.
  llm_completion_json = llm_output.json()
  llm_completion = json.loads(llm_completion_json)
  # Reformat the text in the completions for easier reading in the YAML file.
  for i, generations in enumerate(llm_completion['generations']):
    for j, generation in enumerate(generations):
      text = generation['text']
      llm_completion['generations'][i][j]['text'] = LiteralScalarString(text)

  evaluator_metadata_dict = dict(
    believability_ranking_prompt = LiteralScalarString(believability_ranking_prompt),
    llm_parameters = llm_parameters,
    llm_completion = llm_completion,
  )

  return (
    llm_output,
    evaluator_metadata_dict,
    shuffled_conditions_list,
    ranking_keys_to_condition_keys,
  )


def get_ranking_dict(i, g, ranking_keys_to_condition_keys):
  # The first line contains the ranking string.
  lines = g.message.content.splitlines()
  ranking_str = lines[0]

  # Parse ranking string. The LLM will return the ranking keys
  # separated by commas. Sometimes it includes spaces, which the re
  # below takes into account.
  ranked_keys = re.split(r'\W+', ranking_str)
  # Convert from ranking IDs to condition IDs.
  ranked_condition_keys = [
    ranking_keys_to_condition_keys[rid] for rid in ranked_keys
  ]

  # Save ranking info.
  ranking_dict = dict(
    evaluation_number = i,
    ranking = ranking_str,
    ranked_conditions = ranked_condition_keys,
  )
  return ranking_dict


def get_evaluator_dict(
  evaluator_id,
  interview_question_dict,
  persona_name,
  memory_stream,
):
  # Request LLM completion
  # For debugging, we want to record metadata containing the prompt,
  # LLM parameters, and the raw LLM completion.
  (
    llm_output,
    evaluator_metadata_dict,
    shuffled_conditions,
    ranking_keys_to_condition_keys,
  ) = generate_evaluation(
    interview_question_dict,
    persona_name,
    memory_stream,
  )

  # Parse the rankings.
  ranking_dicts = list()
  for i, g in enumerate(llm_output.generations[0]):
    ranking_dict = get_ranking_dict(i, g, ranking_keys_to_condition_keys)
    ranking_dicts.append(ranking_dict)

  # Save the list of rankinigs.
  evaluator_dict = dict(
    evaluator_id = evaluator_id,
    shuffled_conditions = shuffled_conditions,
    rankings = ranking_dicts,
    evaluator_metadata = evaluator_metadata_dict,
  )
  return evaluator_dict


def get_question_dict(interview_question_dict, persona_name, memory_stream):
  question_id = interview_question_dict['question_id']
  question = interview_question_dict['question']
  evaluator_dicts = list()
  for evaluator_id in ['1. gpt-3.5-turbo-16k']:
    evaluator_dict = get_evaluator_dict(
      evaluator_id,
      interview_question_dict,
      persona_name,
      memory_stream,
    )
    evaluator_dicts.append(evaluator_dict)
        # Save the question and rankings.
  evaluation_question_dict = dict(
    question_id = question_id,
    question = question,
    evaluators = evaluator_dicts,
  )
  return evaluation_question_dict


def get_category_dict(interview_category_dict, persona_name, memory_stream):
  category = interview_category_dict['category']
  interview_question_dicts = interview_category_dict['questions']
  evaluation_question_dicts = list()
  for interview_question_dict in interview_question_dicts:
    evaluation_question_dict = get_question_dict(interview_question_dict, persona_name, memory_stream)
    evaluation_question_dicts.append(evaluation_question_dict)
  # Save the category and list of questions with rankings.
  e_category_dict = dict(
    category = category,
    questions = evaluation_question_dicts,
  )
  return e_category_dict


def get_persona_dict(interview_persona_dict, personas):
  persona_name = interview_persona_dict['persona']
  persona = personas[persona_name]
  memory_stream = persona.format_memory_stream()
  interview_category_dicts = interview_persona_dict['categories']
  evaluation_category_dicts = list()
  for interview_category_dict in interview_category_dicts:
    evaluation_category_dict = get_category_dict(
      interview_category_dict,
      persona_name,
      memory_stream,
    )
    evaluation_category_dicts.append(evaluation_category_dict)
  # Save the persona name and questions organized by categories.
  e_persona_dict = dict(
    persona_name = persona_name,
    categories = evaluation_category_dicts,
  )
  return e_persona_dict


def get_evaluations_dict(personas, interviews):
  interviews_dict = interviews['interviews']
  interview_persona_dicts = interviews_dict['personas']
  evaluation_persona_dicts = list()
  for interview_persona_dict in interview_persona_dicts:
    evaluation_persona_dict = get_persona_dict(interview_persona_dict, personas)
    evaluation_persona_dicts.append(evaluation_persona_dict)
  # Save the evaluations organized by persona.
  evaluations_dict = dict(
    evaluations = dict(personas = evaluation_persona_dicts),
  )

  return evaluations_dict
