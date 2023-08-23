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
from reverie_setup import ReverieTestServer, rs

import reverie
from reverie import ReverieServer
from persona.analysis.interview import interview_persona
from persona.analysis.ablate import (
  ablate_observations_planning_reflection,
  ablate_planning_reflection,
  ablate_reflection,
  is_planning,
  is_reflection,
  is_reflection_error,
)
from persona.analysis.believability import (
  believability_interviews,
  BelievabilityInterviewer,
)
from persona.analysis.believability.believability_questions import (
  get_chat_interaction_counts,
  get_max_chat_interactions,
  get_believability_question_variables,
)
from persona.cognitive_modules.converse import generate_summarize_ideas, generate_next_line
from persona.cognitive_modules.perceive import generate_poig_score
from persona.cognitive_modules.retrieve import new_retrieve
from persona.prompt_template.gpt_structure import get_embedding

import openai.error

import pytest

import jsonpickle

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

import datetime as dt
import functools as ft
import json, pprint, shutil
import random


yaml = YAML()


@pytest.fixture
def questions(rs):
  environment_loc = f"{project_dir}/environment"
  fs_storage = f"{environment_loc}/frontend_server/storage"
  sim_folder = f"{fs_storage}/{rs.sim_code}"
  interview_questions_path = f'{project_dir}/reverie/backend_server/persona/analysis/believability/V1_interview_questions/believability_templates.json'
  believability_dir = f'{sim_folder}/analysis/believability'
  os.makedirs(believability_dir, exist_ok=True)
  with open(interview_questions_path, 'rb') as f:
    templates = json.load(f)
  templates = {'plans':templates['plans']}
  templates['plans'] = {'just_finished_at_1pm': templates['plans']['just_finished_at_1pm']}
  return templates


@pytest.fixture
def interviewer(rs, questions):
  interviewer = BelievabilityInterviewer(
    question_templates=questions,
    personas=rs.personas,
    random_persona_clause="organizing a Valentine's Day party",
    event="a Valentine's Day party",
    random_seed=0,
  )
  return interviewer


### Tests

# Try to make system more deterministic.
random.seed(0)


def test_integration__believability_interviews(rs, questions):
  environment_loc = f"{project_dir}/environment"
  fs_storage = f"{environment_loc}/frontend_server/storage"
  sim_folder = f"{fs_storage}/{rs.sim_code}"
  persona = rs.personas['Isabella Rodriguez']
  personas_to_interview = {persona.name:persona}

  interviews_dict = believability_interviews(
    rs.personas,
    sim_folder,
    random_seed=0,
    personas_to_interview=personas_to_interview,
    question_templates=questions,
  )

  assert 'interviews' in interviews_dict
  assert 'personas' in interviews_dict['interviews']
  assert isinstance(interviews_dict['interviews']['personas'], list)
  first_persona_dict = interviews_dict['interviews']['personas'][0]
  assert 'persona' in first_persona_dict
  assert persona.name == first_persona_dict['persona']
  assert 'categories' in first_persona_dict
  assert isinstance(first_persona_dict['categories'], list)
  first_category_dict = first_persona_dict['categories'][0]
  assert 'category' in first_category_dict
  assert 'plans' == first_category_dict['category']
  assert 'questions' in first_category_dict
  assert isinstance(first_category_dict['questions'], list)
  first_question_dict = first_category_dict['questions'][0]
  assert 'question_id' in first_question_dict
  assert first_question_dict['question_id'] == 'just_finished_at_1pm'
  assert 'question' in first_question_dict
  assert 'conditions' in first_question_dict
  assert isinstance(first_question_dict['conditions'], list)
  assert len(first_question_dict['conditions']) == 4
  first_condition_dict = first_question_dict['conditions'][0]
  assert 'condition' in first_condition_dict
  assert 'response' in first_condition_dict
  assert 'summarized_idea' in first_condition_dict
  assert first_condition_dict['condition'] == 'no_observation_no_reflection_no_planning'


def test_integration__BelievabilityInterviewer__generate_interviews_dict(rs, questions):
  interviewer = BelievabilityInterviewer(
    question_templates=questions,
    personas=rs.personas,
    random_persona_clause="organizing a Valentine's Day party",
    event="a Valentine's Day party",
    random_seed=0,
  )
  persona = rs.personas['Isabella Rodriguez']
  personas_to_interview = {persona.name:persona}
  interviews_dict = interviewer.generate_interviews_dict(personas=personas_to_interview)

  assert 'interviews' in interviews_dict
  assert 'personas' in interviews_dict['interviews']
  assert isinstance(interviews_dict['interviews']['personas'], list)
  first_persona_dict = interviews_dict['interviews']['personas'][0]
  assert 'persona' in first_persona_dict
  assert persona.name == first_persona_dict['persona']
  assert 'categories' in first_persona_dict
  assert isinstance(first_persona_dict['categories'], list)
  first_category_dict = first_persona_dict['categories'][0]
  assert 'category' in first_category_dict
  assert 'plans' == first_category_dict['category']
  assert 'questions' in first_category_dict
  assert isinstance(first_category_dict['questions'], list)
  first_question_dict = first_category_dict['questions'][0]
  assert 'question_id' in first_question_dict
  assert first_question_dict['question_id'] == 'just_finished_at_1pm'
  assert 'question' in first_question_dict
  assert 'conditions' in first_question_dict
  assert isinstance(first_question_dict['conditions'], list)
  assert len(first_question_dict['conditions']) == 4
  first_condition_dict = first_question_dict['conditions'][0]
  assert 'condition' in first_condition_dict
  assert 'response' in first_condition_dict
  assert 'summarized_idea' in first_condition_dict
  assert first_condition_dict['condition'] == 'no_observation_no_reflection_no_planning'


def test_brainstorm__freeze_thaw_ablate_interview(rs):
  persona = rs.personas['Isabella Rodriguez']
  persona_frozen = jsonpickle.encode(persona, indent=2)
  persona_thawed = jsonpickle.decode(persona_frozen)

  ablate_reflection(persona_thawed)
  question = 'Give an introduction of yourself.'

  _,_, response, current_convo = interview_persona(
    persona=persona_thawed,
    message=question
  )
  log.debug(
    f'''
--- Interview question:
Question: {question}
Response:
{response}
'''
  )


# Helper function to count different types of thought memories.
def count_thoughts(
  id_to_node
):
  thoughts = {
    key:node
    for key,node in id_to_node.items()
    if node.type == 'thought'
  }
  plans = {
    key:node
    for key,node in thoughts.items()
    if node.filling is None
  }
  reflections = {
    key:node
    for key,node in thoughts.items()
    if isinstance(node.filling, list)
  }
  errors = {
    key:node
    for key,node in thoughts.items()
    if isinstance(node.filling, str)
  }

  thought_ct = len(thoughts)
  plan_ct = len(plans)
  reflection_ct = len(reflections)
  error_ct = len(errors)

  return thought_ct, plan_ct, reflection_ct, error_ct


def test_brainstorm__freeze_thaw_ablate_count(rs):
  persona = rs.personas['Isabella Rodriguez']

  # Count persona's thoughts before ablation.
  (
    orig_thought_ct,
    orig_plan_ct,
    orig_reflection_ct,
    orig_error_ct,
  ) = count_thoughts(persona.a_mem.id_to_node)

  persona_frozen = jsonpickle.encode(persona, indent=2)
  persona_thawed = jsonpickle.decode(persona_frozen)
  ablate_planning_reflection(persona_thawed)

  # Count original persona's thoughts after ablation of thawed persona. Counts shouldn't change.
  (
    new_thought_ct,
    new_plan_ct,
    new_reflection_ct,
    new_error_ct,
  ) = count_thoughts(persona.a_mem.id_to_node)

  # Count thawed persona's thoughts after ablation. Counts should differ.
  (
    thawed_thought_ct,
    thawed_plan_ct,
    thawed_reflection_ct,
    thawed_error_ct,
  ) = count_thoughts(persona_thawed.a_mem.id_to_node)

  # Counts of original persona's thoughts shouldn't change.
  assert new_thought_ct == orig_thought_ct
  assert new_plan_ct == orig_plan_ct
  assert new_reflection_ct == orig_reflection_ct
  assert new_error_ct == orig_error_ct

  # After ablation, counts of thawed persona's thoughts should differ.
  assert thawed_thought_ct != orig_thought_ct
  assert thawed_plan_ct != orig_plan_ct
  assert thawed_reflection_ct != orig_reflection_ct
  assert thawed_error_ct != orig_error_ct


def test_brainstorm__jsonpickle_persona(rs):
  persona = rs.personas['Isabella Rodriguez']
  # Ensure can encode without raising exceptions.
  persona_frozen_1 = jsonpickle.encode(persona, indent=2)
  persona_frozen_2 = jsonpickle.encode(persona, indent=2)
  persona_thawed = jsonpickle.decode(persona_frozen_1)
  persona_frozen_3 = jsonpickle.encode(persona_thawed, indent=2)

  #with open(f'{project_dir}/tests/persona_frozen_1.json', 'w') as f:
  #  f.write(persona_frozen_1)
  #with open(f'{project_dir}/tests/persona_frozen_2.json', 'w') as f:
  #  f.write(persona_frozen_2)
  #with open(f'{project_dir}/tests/persona_frozen_3.json', 'w') as f:
  #  f.write(persona_frozen_3)


def test_brainstorm__prototype__interview_with_ablations__reflection(rs):
  persona = rs.personas['Isabella Rodriguez']
  ablate_reflection(persona)
  question = 'Give an introduction of yourself.'

  _,_, response, current_convo = interview_persona(
    persona=persona,
    message=question
  )
  log.debug(
    f'''
--- Interview question:
Question: {question}
Response:
{response}
'''
  )


def test_brainstorm__prototype__interview_with_ablations__planning_reflection(rs):
  persona = rs.personas['Isabella Rodriguez']
  ablate_planning_reflection(persona)
  question = 'Give an introduction of yourself.'

  _,_, response, current_convo = interview_persona(
    persona=persona,
    message=question
  )
  log.debug(
    f'''
--- Interview question:
Question: {question}
Response:
{response}
'''
  )


def test_brainstorm__prototype__interview_with_ablations__observation_planning_reflection(rs):
  persona = rs.personas['Isabella Rodriguez']
  ablate_observations_planning_reflection(persona)
  question = 'Give an introduction of yourself.'

  _,_, response, current_convo = interview_persona(
    persona=persona,
    message=question
  )
  log.debug(
    f'''
--- Interview question:
Question: {question}
Response:
{response}
'''
  )


# Below I'm examining the initial event, thought, and chat memories of Isabella
# Rodriguez. I'm trying to find a set of memories with the least information
# possible.
#
# Reason: For full ablation I tried interviewing an agent after emptying all
# memory structures, but this causes exception `ValueError: min() arg is an
# empty sequence`:
#  - `test_brainstorm__analysis.py:102:test_brainstorm__prototype__persona_ablations__observations_planning_reflections()`
#  - `interview.py:35:interview_persona()`
#  - `retrieve.py:242:new_retrieve()`
#  - `retrieve.py:93:normalize_dict_floats()`
#
#  The cause:
#  - `new_retrieve()` calls `extract_recency()` which returns an empty
#    dictionary. This empty dictionary is passed to `normalize_dict_floats()`.
#    `normalize_dict_floats()` tries to find the maximum of the empty
#    dictionary's values, but the value list is empty, so the `max` function
#    raises the exception. 
#  - `extract_recency()` produces the empty dictionary because it is passed an
#    empty `nodes` list  as its second argument.
#  - `new_retrieve()` constructs this empty nodes list by iterating over
#    `persona.a_mem.seq_event + persona.a_mem.seq_thought`, both of which I've
#    emptied.
#
# Result:
#
# ```python
# Initial event memory:
# { 'created': datetime.datetime(2023, 2, 13, 0, 0),
#   'depth': 0,
#   'description': 'Isabella Rodriguez is idle',
#   'embedding_key': 'Isabella Rodriguez is idle',
#   'expiration': None,
#   'filling': [],
#   'keywords': {'Isabella Rodriguez', 'idle'},
#   'last_accessed': datetime.datetime(2023, 2, 13, 0, 0),
#   'node_count': 1,
#   'node_id': 'node_1',
#   'object': 'idle',
#   'poignancy': 1,
#   'predicate': 'is',
#   'subject': 'Isabella Rodriguez',
#   'type': 'event',
#   'type_count': 1}
# Initial thought memory:
# { 'created': datetime.datetime(2023, 2, 13, 0, 0),
#   'depth': 1,
#   'description': "This is Isabella Rodriguez's plan for Monday February 13: "
#                  'wake up and complete the morning routine at 6:00 am, travel '
#                  'to Hobbs Cafe at 7:00 am, open up shop at 8:00 am, greet '
#                  'customers and work at the counter until 8 pm, buy party '
#                  "materials for the Valentine's Day party at the cafe from "
#                  '9:00 am to 10:00 am, have lunch at 12:00 pm, take a short '
#                  "nap from 2 to 4 pm, plan the Valentine's Day Party in the "
#                  'afternoon.',
#   'embedding_key': "This is Isabella Rodriguez's plan for Monday February 13: "
#                    'wake up and complete the morning routine at 6:00 am, '
#                    'travel to Hobbs Cafe at 7:00 am, open up shop at 8:00 am, '
#                    'greet customers and work at the counter until 8 pm, buy '
#                    "party materials for the Valentine's Day party at the cafe "
#                    'from 9:00 am to 10:00 am, have lunch at 12:00 pm, take a '
#                    "short nap from 2 to 4 pm, plan the Valentine's Day Party "
#                    'in the afternoon.',
#   'expiration': datetime.datetime(2023, 3, 15, 0, 0),
#   'filling': None,
#   'keywords': {'plan'},
#   'last_accessed': datetime.datetime(2023, 2, 13, 0, 0),
#   'node_count': 7,
#   'node_id': 'node_7',
#   'object': 'Monday February 13',
#   'poignancy': 5,
#   'predicate': 'plan',
#   'subject': 'Isabella Rodriguez',
#   'type': 'thought',
#   'type_count': 1}
# Initial chat memory:
# { 'created': datetime.datetime(2023, 2, 13, 11, 22, 40),
#   'depth': 0,
#   'description': 'conversing about a conversation about Isabella inviting '
#                  "Klaus to her Valentine's Day party at Hobbs Cafe on February "
#                  '14th, 2023 from 5pm to 7pm.',
#   'embedding_key': 'conversing about a conversation about Isabella inviting '
#                    "Klaus to her Valentine's Day party at Hobbs Cafe on "
#                    'February 14th, 2023 from 5pm to 7pm.',
#   'expiration': None,
#   'filling': [ [ 'Isabella Rodriguez',
#                  'Hi Klaus! How are you enjoying your meal? I wanted to let '
#                  "you know that I'm planning a Valentine's Day party at Hobbs "
#                  'Cafe on February 14th, 2023 from 5pm to 7pm. I would love '
#                  'for you to join us!'],
#                [ 'Klaus Mueller',
#                  "Oh, hi Isabella! I'm doing well, thank you. The meal is "
#                  "delicious as always. A Valentine's Day party sounds fun. I'd "
#                  'love to join! Thank you for inviting me.']],
#   'keywords': {'Klaus Mueller', 'Isabella Rodriguez'},
#   'last_accessed': datetime.datetime(2023, 2, 13, 11, 22, 40),
#   'node_count': 287,
#   'node_id': 'node_287',
#   'object': 'Klaus Mueller',
#   'poignancy': 4,
#   'predicate': 'chat with',
#   'subject': 'Isabella Rodriguez',
#   'type': 'chat',
#   'type_count': 1}
# ```
def test_brainstorm__persona_initial_memories(rs):
  persona = rs.personas['Isabella Rodriguez']
  # Examine initial event, thought, and chat memories.
  initial_event_memory = persona.a_mem.seq_event[-1]
  second_event_memory = persona.a_mem.seq_event[-2]
  initial_thought_memory = persona.a_mem.seq_thought[-1]
  initial_chat_memory = persona.a_mem.seq_chat[-1]
  log.debug(
    f'''
Initial event memory:
{pprint.pformat(initial_event_memory.__dict__, indent=2)}
Second event memory:
{pprint.pformat(second_event_memory.__dict__, indent=2)}
Initial thought memory:
{pprint.pformat(initial_thought_memory.__dict__, indent=2)}
Initial chat memory:
{pprint.pformat(initial_chat_memory.__dict__, indent=2)}
'''
  )

  # Examine whether id_to_node maps node ids to nodes (it does.)
  initial_event_lookup = persona.a_mem.id_to_node[initial_event_memory.node_id]

  # Construct a minimal kw_to_event dict that refers to just the initial event
  # memory.
  narrowed_kw_to_event = {
    k:[node for node in nodes if node.node_id == initial_event_memory.node_id]
    for k,nodes in persona.a_mem.kw_to_event.items()
    if initial_event_memory.node_id in [
      node.node_id
      for node in nodes
    ]
  }
  log.debug(
    f'''
Initial event lookup:
{pprint.pformat(initial_event_lookup.__dict__, indent=2)}
Narrowed_kw_to_event:
{pprint.pformat(narrowed_kw_to_event, indent=2)}
'''
  )



@pytest.mark.parametrize(
  'persona_name',
  ['Isabella Rodriguez', 'Maria Lopez', 'Klaus Mueller']
)
def test_brainstorm__prototype__associative_memory_filters(persona_name, rs):
  persona = rs.personas[persona_name]
  id_to_node = persona.a_mem.id_to_node

  events = {
    key:node
    for key,node in id_to_node.items()
    if node.type == 'event'
  }
  chats = {
    key:node
    for key,node in id_to_node.items()
    if node.type == 'chat'
  }
  thoughts = {
    key:node
    for key,node in id_to_node.items()
    if node.type == 'thought'
  }

  assert (len(events) + len(chats) + len(thoughts)) == len(id_to_node)

  plans = {
    key:node
    for key,node in id_to_node.items()
    if is_planning(node)
  }
  reflections = {
    key:node
    for key,node in id_to_node.items()
    if is_reflection(node)
  }
  errors = {
    key:node
    for key,node in id_to_node.items()
    if is_reflection_error(node)
  }

  assert (len(plans) + len(reflections) + len(errors)) == len(thoughts)
  assert (len(plans) + len(reflections) + len(errors)) == len(persona.a_mem.seq_thought)



# Parameterize `...filter_associative_memory` with the names of the personas.
#
@pytest.mark.parametrize(
  'persona_name',
  ['Isabella Rodriguez', 'Maria Lopez', 'Klaus Mueller']
)
def test_brainstorm__filter_associative_memory(persona_name, rs):
  persona = rs.personas[persona_name]
  id_to_node = persona.a_mem.id_to_node

  events = {
    key:node
    for key,node in id_to_node.items()
    if node.type == 'event'
  }
  chats = {
    key:node
    for key,node in id_to_node.items()
    if node.type == 'chat'
  }
  thoughts = {
    key:node
    for key,node in id_to_node.items()
    if node.type == 'thought'
  }

  assert (len(events) + len(chats) + len(thoughts)) == len(id_to_node)

  # There are three types of thought memories:
  # - Plans: per the Park et al generative-agents paper, the persona starts
  #   each day by building a broad-strokes plan for the day, which is then
  #   broken down to hour-by-hour, then further broken down to 5-10 minute
  #   blocks.
  #
  #   Plan memories are characterized by:
  #   - `node.type == 'thought'`
  #   - `node.filling is None
  #   - `node.keywords == ['plan']
  #   - `node.depth == 1`
  #
  #   The first and second characteristics are currently sufficient to identify
  #   plan memories. The third is not sufficient, as it is possible for other
  #   kinds of memories to have the single keyword 'plan'. The fourth is not
  #   sufficient, as reflections can also have `depth == 1`.
  #
  # - Reflections: per the paper, reflections are higher-order memories
  #   generated by a language model tasked with generating insights about
  #   recent memories. Reflections cite lower-order memories as evidence.
  #   Currently, event and chat memories are treated as zeroth-order memories,
  #   and plans as first-order memories. The order of a reflection is then one
  #   more than the highest order evidence memory. The order of a memory is
  #   stored in the `depth` attribute.
  #
  #   Reflection memories are characterized by:
  #  - `node.type == 'thought'`
  #  - `isinstance(node.filling, list)`
  #  - `node.depth >= 1`
  #
  #  Currently, the first and second characteristics are sufficient to identify
  #  reflection memories. The third is not sufficient, as both plans and
  #  reflections can be first-order memories.
  #
  # - Errors: these are generated in the `persona.cognitive_modules.reflect`
  #   module when a Python exception is encountered in the
  #   `generate_insights_and_evidence(...)` function. This appears to be the
  #   only code in Reveries that generates this kind of memory. It can be
  #   thought of as a failed reflection. These errors are characterized by:
  #   - `node.type == 'thought'
  #   - `node.filling == 'node_1' # Notably, this is a string, not a list.
  #   - `node.description == 'this is blank'
  #   - `node.depth == 1`
  #
  #   Currently the first and second characteristics are sufficient to identify
  #   these reflection errors. I would hesitate to use `filling == 'node_1'` as
  #   criterion, but rather `isinstance(filling, str)`.
  #
  #   It might be a good idea to ask Park (@joonspk-research) about this. I
  #   suspect he's in the process of developing some coding idea here, so these
  #   reflection errors might either go away in future code, or the criteria
  #   for identifying them might change.
  #
  plans = {
    key:node
    for key,node in thoughts.items()
    if node.filling is None
  }
  reflections = {
    key:node
    for key,node in thoughts.items()
    if isinstance(node.filling, list)
  }
  errors = {
    key:node
    for key,node in thoughts.items()
    if isinstance(node.filling, str)
  }

  assert (len(plans) + len(reflections) + len(errors)) == len(thoughts)
  assert (len(plans) + len(reflections) + len(errors)) == len(persona.a_mem.seq_thought)



# Parameterize `...pose_one_believability_question` with each of the
# believability questions listed in `believability_templates.json`.
#
interview_questions_path = f'{project_dir}/reverie/backend_server/persona/analysis/believability/V1_interview_questions/believability_templates.json'
with open(interview_questions_path, 'rb') as f:
  believability_questions = json.load(f)
question_templates = [
  (
    believability_area,
    topic,
    template
  )
  for believability_area, templates in list(believability_questions.items())[:1]
  for topic, template in list(templates.items())[:1]
]
@pytest.mark.parametrize(
  'area,topic,question_template',
  question_templates
)
def test_brainstorm__pose_one_believability_question(area, topic, question_template, rs):
  langchain.llm_cache.hit_miss_tracker = []
  persona_names = list(rs.personas.keys())
  persona = rs.personas['Isabella Rodriguez']
  question_variables = get_believability_question_variables(
    persona=persona,
    personas=rs.personas,
    random_persona_clause = "organizing a Valentine's Day party",
    event = "a Valentine's Day party",
    # Get a deterministic random number generator by seeding with 0.
    random_seed = 0,
  )
  question = question_template.format_map(question_variables)
  _,_, response, current_convo = interview_persona(
    persona=persona,
    message=question,
  )
  hit_miss = langchain.llm_cache.hit_miss_tracker
  langchain.llm_cache.hit_miss_tracker = None
  misses = [hm for hm in hit_miss if hm['hit_miss'] == 'MISS']
  #response = '(not requested)'
  log.debug(
    f'''

--- Interview question:
Area: {area}
Topic: {topic}
Question: {question}
Response:
{response}

LangChain cache misses:
{misses}
'''
  )
  assert not misses


def test_brainstorm__get_believability_question_variables(rs):
  persona_names = list(rs.personas.keys())
  persona = rs.personas['Isabella Rodriguez']
  question_variables = get_believability_question_variables(
    persona=persona,
    personas=rs.personas,
    random_persona_clause = "organizing a Valentine's Day party",
    event = "a Valentine's Day party",
    # Get a deterministic random number generator by seeding with 0.
    random_seed = 0,
  )

  interview_questions_path = f'{project_dir}/reverie/backend_server/persona/analysis/believability/V1_interview_questions/believability_templates.json'
  with open(interview_questions_path, 'rb') as f:
    believability_questions = json.load(f)
  questions_lines = []
  for believability_area, question_templates in believability_questions.items():
    questions_lines.append(f'Believability area: {believability_area}:')
    for topic, question_template in question_templates.items():
      question = question_template.format_map(question_variables)
      questions_lines.append(f'  Topic: {topic}: {question}')
  questions_text = '\n'.join(questions_lines)
  log.debug(
    f'''
{questions_text}
'''
  )


def test_brainstorm__prototype__interview_question_formatting(rs):
  # Get a deterministic random number generator by seeding with 0.
  rng = random.Random(0)

  persona_names = list(rs.personas.keys())
  persona = rs.personas['Isabella Rodriguez']
  persona_names.remove(persona.name)

  random_persona_name = rng.choice(persona_names)
  random_persona_1 = rs.personas[random_persona_name]
  persona_names.remove(random_persona_1.name)
  random_persona_name = rng.choice(persona_names)
  random_persona_2 = rs.personas[random_persona_name]
  persona_names.remove(random_persona_2.name)

  max_chats, max_dialog_exchanges = get_max_chat_interactions(persona)
  well_known_persona_name = max_dialog_exchanges[0]

  question_variables = dict(
    random_persona_name_1 = random_persona_1.name,
    random_persona_name_2 = random_persona_2.name,
    random_persona_clause = "organizing a Valentine's Day party",
    event = "a Valentine's Day party",
    well_known_persona_name = well_known_persona_name,
  )

  interview_questions_path = f'{project_dir}/reverie/backend_server/persona/analysis/believability/V1_interview_questions/believability_templates.json'
  with open(interview_questions_path, 'rb') as f:
    believability_questions = json.load(f)
  questions_lines = []
  for believability_area, question_templates in believability_questions.items():
    questions_lines.append(f'Believability area: {believability_area}:')
    for topic, question_template in question_templates.items():
      question = question_template.format_map(question_variables)
      questions_lines.append(f'  Topic: {topic}: {question}')
  questions_text = '\n'.join(questions_lines)
  log.debug(
    f'''
{questions_text}
'''
  )


def test_brainstorm__prototype__get_max_chat_interactions(rs):
  persona = rs.personas['Isabella Rodriguez']
  max_chats, max_dialog_exchanges = get_max_chat_interactions(persona)
  log.debug(
    f'''
Max chats: {max_chats}
Max dialog exchanges: {max_dialog_exchanges}
'''
  )


def test_brainstorm__prototype__get_chat_interaction_counts(rs):
  persona = rs.personas['Isabella Rodriguez']
  chat_counts, dialog_exchange_counts = get_chat_interaction_counts(persona)

  max_chats = max(chat_counts.items(), key=lambda x: x[1])
  max_dialog_exchanges = max(dialog_exchange_counts.items(), key=lambda x: x[1])

  chat_counts_lines = []
  for participant, count in chat_counts.items():
    line = f'    {participant}: {count}'
    chat_counts_lines.append(line)
  chat_counts_text = '\n'.join(chat_counts_lines)

  dialog_exchange_counts_lines = []
  for participant, count in dialog_exchange_counts.items():
    line = f'    {participant}: {count}'
    dialog_exchange_counts_lines.append(line)
  dialog_exchange_counts_text = '\n'.join(dialog_exchange_counts_lines)

  log.debug(
    f'''
Interaction counts:
  Chat counts:
{chat_counts_text}
  Dialog exchange counts:
{dialog_exchange_counts_text}

Max chats: {max_chats}
Max dialog exchanges: {max_dialog_exchanges}
'''
  )


def test_brainstorm__persona_chat_memory(rs):
  persona = rs.personas['Isabella Rodriguez']
  chat_counts = dict()
  dialog_exchange_counts = dict()
  for count, event in enumerate(persona.a_mem.seq_chat):
    participant = event.object

    # Increment count of chats with chat participant.
    chat_count = chat_counts.get(participant, 0)
    chat_count += 1
    chat_counts[participant] = chat_count

    description = event.description
    created_at = event.created.strftime('%B %d, %Y, %H%M%S')
    lines = []
    for row in event.filling:
      speaker, dialog = row

      if speaker != persona.name:
        # Increment count of chats with chat participant.
        dialog_exchange_count = dialog_exchange_counts.get(speaker, 0)
        dialog_exchange_count += 1
        dialog_exchange_counts[speaker] = dialog_exchange_count

      line = f'{speaker}: {dialog}'
      lines.append(line)
    dialog_history = '\n\n'.join(lines)
    log.debug(
      f'''
Chat:
  {participant=}
  {description=}
  {created_at=}

  Dialog history:

  {dialog_history}
'''
  )

  chat_counts_lines = []
  for participant, count in chat_counts.items():
    line = f'    {participant}: {count}'
    chat_counts_lines.append(line)
  chat_counts_text = '\n'.join(chat_counts_lines)

  dialog_exchange_counts_lines = []
  for participant, count in dialog_exchange_counts.items():
    line = f'    {participant}: {count}'
    dialog_exchange_counts_lines.append(line)
  dialog_exchange_counts_text = '\n'.join(dialog_exchange_counts_lines)

  log.debug(
    f'''
Interaction counts:
  Chat counts:
{chat_counts_text}
  Dialog exchange counts:
{dialog_exchange_counts_text}
'''
  )


def test_brainstorm__interview_question_file(rs):
  interview_questions_path = f'{project_dir}/reverie/backend_server/persona/analysis/believability/V1_interview_questions/believability_templates.json'
  with open(interview_questions_path, 'rb') as f:
    believability_questions = json.load(f)
  log.debug(
    f'''
{believability_questions=}
'''
  )
  questions_lines = []
  for believability_area, questions in believability_questions.items():
    questions_lines.append(f'{believability_area}:')
    for topic, question in questions.items():
      questions_lines.append(f'  {topic}:')
      questions_lines.append(f'    {question}')

  questions_text = '\n\n'.join(questions_lines)
  log.debug(
    f'''
{questions_text}
'''
  )


def test_regression__interview_persona__context_length_exceeded(rs):
  persona = rs.personas['Isabella Rodriguez']
  curr_convo = []
  interviewer = "Interviewer"
  message = "How are you?"
  context_length_exceeded = False
  try:
    _,_, response, curr_convo = interview_persona(
      persona=persona,
      message=message,
      curr_convo=curr_convo,
      interviewer=interviewer,
      n_count=30,
    )
  except openai.error.InvalidRequestError as e:
    if e.code == 'context_length_exceeded':
      context_length_exceeded = True
    log.warning(f'OpenAI InvalidRequestError: exception: {e}')
  assert not context_length_exceeded, "OpenAI context length exceeded"

  convo_lines = '\n\n'.join(f'{speaker.upper()}: {line}' for (speaker, line) in curr_convo)
  log.debug(
    f'''
*** Current convo:

{convo_lines}
''')


def test_brainstorm__interview_persona(rs):
  persona = rs.personas['Isabella Rodriguez']
  curr_convo = []
  interviewer = "Interviewer"
  message = "How are you?"
  _,_, response, curr_convo = interview_persona(
    persona=persona,
    message=message,
    curr_convo=curr_convo,
    interviewer=interviewer,
  )

  convo_lines = '\n\n'.join(f'{speaker.upper()}: {line}' for (speaker, line) in curr_convo)
  log.debug(
    f'''
*** Current convo:

{convo_lines}
''')


def test_brainstorm__prototype_interview_persona(rs):
  persona = rs.personas['Isabella Rodriguez']
  curr_convo = []
  interviewer = "Interviewer"
  message = "How are you?"
  retrieved = new_retrieve(
    persona=persona,
    focal_points=[message],
    n_count=30,
    weights=(1.,1.,1.),
  )[message]
  summarized_idea = generate_summarize_ideas(persona, retrieved, message)
  curr_convo += [[interviewer, message]]
  response = generate_next_line(persona, interviewer, curr_convo, summarized_idea)
  curr_convo += [[persona.scratch.name, response]]

  convo_lines = '\n\n'.join(f'{speaker.upper()}: {line}' for (speaker, line) in curr_convo)
  log.debug(
    f'''
*** Summarized idea:
{summarized_idea}

*** Next line:
{response}

*** Current convo:

{convo_lines}
''')
