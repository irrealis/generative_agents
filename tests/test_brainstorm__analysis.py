import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from dotenv import find_dotenv, load_dotenv
env_path = find_dotenv()
load_dotenv(env_path)

import os
import random
import sys

project_dir = os.path.dirname(env_path)
log.debug(f'{project_dir=}')

sys.path.insert(0, os.path.abspath(f"{project_dir}/reverie/backend_server"))
sys.path.insert(0, os.path.abspath(f"{project_dir}/../datascience"))


import utils

import reverie
from reverie import ReverieServer
from persona.analysis.interview import interview_persona
from persona.analysis.believability_questions import get_chat_interaction_counts, get_max_chat_interactions, get_believability_question_variables
from persona.persona import Persona
from persona.cognitive_modules.retrieve import extract_recency, extract_importance, extract_relevance, new_retrieve, normalize_dict_floats, top_highest_x_values
from persona.cognitive_modules.converse import generate_summarize_ideas, generate_next_line

from irrealis.generative_agents.test_tools import *

import langchain
from langchain.cache import SQLiteCache, RETURN_VAL_TYPE

import openai.error

import pytest

import datetime as dt
import json, shutil


#langchain.llm_cache = SQLiteCache_ForTests(database_path=".langchain.db")
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']='Park Generative Agents'
tags = ['test', 'brainstorm']
db_url = "postgresql+psycopg2://kaben:{DB_PASS}@localhost:5432/datascience".format(
  DB_PASS=os.getenv('DB_PASS')
)

# To the user: set with your OpenAI API key.


class ReverieTestServer(ReverieServer):
  '''
  Version of ReverieServer for test fixtures.
  '''
  def __init__(self, fork_sim_code, sim_code, predelete=False):
    '''
    Overrides ReverieServer.__init__() to optionally predelete sim directory

    Predelete is for test purposes.
    '''
    sim_path=f'{reverie.fs_storage}/{sim_code}'
    if predelete and os.path.exists(sim_path):
      shutil.rmtree(sim_path)
    super().__init__(fork_sim_code=fork_sim_code, sim_code=sim_code)


@pytest.fixture
def rs():
  fork_sim_code = 'July1_the_ville_isabella_maria_klaus-step-3-20'
  sim_code = 'test-simulation'
  return ReverieTestServer(
    fork_sim_code=fork_sim_code,
    sim_code=sim_code,
    predelete=True,
  )



### Tests


def test_brainstorm__pose_believability_questions(rs):
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

  interview_questions_path = f'{project_dir}/reverie/backend_server/persona/analysis/V1_interview_questions/believability_templates.json'
  with open(interview_questions_path, 'rb') as f:
    believability_questions = json.load(f)
  for believability_area, question_templates in believability_questions.items():
    for topic, question_template in question_templates.items():
      question = question_template.format_map(question_variables)
      response, current_convo = interview_persona(
        persona=persona,
        message=question,
      )
      log.debug(
        f'''

--- Interview question:
Area: {believability_area}
Topic: {topic}
Question: {question}
Response:
{response}
'''
  )


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

  interview_questions_path = f'{project_dir}/reverie/backend_server/persona/analysis/V1_interview_questions/believability_templates.json'
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

  interview_questions_path = f'{project_dir}/reverie/backend_server/persona/analysis/V1_interview_questions/believability_templates.json'
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
  interview_questions_path = f'{project_dir}/reverie/backend_server/persona/analysis/V1_interview_questions/believability_templates.json'
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
    response, curr_convo = interview_persona(
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
  response, curr_convo = interview_persona(
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
