import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from dotenv import find_dotenv, load_dotenv
env_path = find_dotenv()
load_dotenv(env_path)

import os
import sys

project_dir = os.path.dirname(env_path)
log.debug(f'{project_dir=}')

sys.path.insert(0, os.path.abspath(f"{project_dir}/reverie/backend_server"))
sys.path.insert(0, os.path.abspath(f"{project_dir}/../datascience"))


import utils

import reverie
from reverie import ReverieServer
from persona.persona import Persona
from persona.cognitive_modules.retrieve import extract_recency, extract_importance, extract_relevance, normalize_dict_floats, top_highest_x_values
from persona.cognitive_modules.converse import generate_summarize_ideas, generate_next_line

from irrealis.generative_agents.test_tools import *

import langchain
from langchain.cache import SQLiteCache, RETURN_VAL_TYPE

import pytest

import datetime as dt
import shutil


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


def retrieve(persona, focal_points, n_count=30):
  """
  Given the current persona and focal points (focal points are events or
  thoughts for which we are retrieving), we retrieve a set of nodes for each
  of the focal points and return a dictionary.
  Example input:
    persona = <persona> object
    focal_points = ["How are you?", "Jane is swimming in the pond"]
  """
  # <retrieved> is the main dictionary that we are returning
  retrieved = dict()
  for focal_pt in focal_points:
    # Getting all nodes from the agent's memory (both thoughts and events) and
    # sorting them by the datetime of creation.
    # You could also imagine getting the raw conversation, but for now. 
    nodes = [[i.last_accessed, i]
              for i in persona.a_mem.seq_event + persona.a_mem.seq_thought
              if "idle" not in i.embedding_key]
    nodes = sorted(nodes, key=lambda x: x[0])
    nodes = [i for created, i in nodes]

    # Calculating the component dictionaries and normalizing them.
    recency_out = extract_recency(persona, nodes)
    recency_out = normalize_dict_floats(recency_out, 0, 1)
    importance_out = extract_importance(persona, nodes)
    importance_out = normalize_dict_floats(importance_out, 0, 1)
    relevance_out = extract_relevance(persona, nodes, focal_pt)
    relevance_out = normalize_dict_floats(relevance_out, 0, 1)

    # Computing the final scores that combines the component values. 
    # Note to self: test out different weights. [1, 1, 1] tends to work
    # decently, but in the future, these weights should likely be learned, 
    # perhaps through an RL-like process.
    gw = [1, 1, 1]
    master_out = dict()
    for key in recency_out.keys():
      master_out[key] = (persona.scratch.recency_w*recency_out[key]*gw[0]
                     + persona.scratch.relevance_w*relevance_out[key]*gw[1]
                     + persona.scratch.importance_w*importance_out[key]*gw[2])

    master_out = top_highest_x_values(master_out, len(master_out.keys()))
    for key, val in master_out.items():
      print (persona.a_mem.id_to_node[key].embedding_key, val)
      print (persona.scratch.recency_w*recency_out[key]*1,
             persona.scratch.relevance_w*relevance_out[key]*1,
             persona.scratch.importance_w*importance_out[key]*1)

    # Extracting the highest x values.
    # <master_out> has the key of node.id and value of float. Once we get the 
    # highest x values, we want to translate the node.id into nodes and return
    # the list of nodes.
    master_out = top_highest_x_values(master_out, n_count)
    master_nodes = [persona.a_mem.id_to_node[key]
                    for key in list(master_out.keys())]

    for n in master_nodes:
      n.last_accessed = persona.scratch.curr_time
    retrieved[focal_pt] = master_nodes

  return retrieved


def interview_persona(persona, message, curr_convo=None, interviewer=None, n_count=None):
  if curr_convo is None:
    curr_convo = []
  if interviewer is None:
    interviewer = 'Interviewer'

  retrieved = retrieve(persona, [message], 30)[message]
  summarized_idea = generate_summarize_ideas(persona, retrieved, message)
  curr_convo += [[interviewer, message]]
  response = generate_next_line(persona, interviewer, curr_convo, summarized_idea)
  curr_convo += [[persona.scratch.name, response]]

  return response, curr_convo


### Tests


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
  retrieved = retrieve(persona, [message], 30)[message]
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
