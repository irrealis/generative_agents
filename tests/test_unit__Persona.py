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

from concept_nodes_setup import concept_nodes
from reverie_setup import ReverieTestServer, rs

from persona.persona import Persona

### Tests


def test_unit__Persona__format_plan_thought(rs, concept_nodes):
  persona = rs.personas['Isabella Rodriguez']
  formatted_thought = persona.format_plan_thought(concept_nodes['plan_thought'])
  expected_formatted_thought = "- At 00:00:00: Isabella Rodriguez is thinking about what to do. This is Isabella Rodriguez's plan for Monday February 13: wake up and complete the morning routine at 6:00 am, travel to Hobbs Cafe at 7:00 am, open up shop at 8:00 am, greet customers and work at the counter until 8 pm, buy party materials for the Valentine's Day party at the cafe from 9:00 am to 10:00 am, have lunch at 12:00 pm, take a short nap from 2 to 4 pm, plan the Valentine's Day Party in the afternoon."
  assert formatted_thought == expected_formatted_thought


def test_unit__Persona__format_bootstrap_or_reflection_thought__bootstrap(rs, concept_nodes):
  persona = rs.personas['Isabella Rodriguez']
  formatted_thought = persona.format_bootstrap_or_reflection_thought(concept_nodes['bootstrap_thought'])
  expected_formatted_thought = "- At 00:00:20: Isabella Rodriguez reflects that Isabella Rodriguez is excited to be planning a Valentine's Day party at Hobbs Cafe on February 14th from 5pm and is eager to invite everyone to attend the party."
  assert formatted_thought == expected_formatted_thought


def test_unit__Persona__format_chat_event(rs, concept_nodes):
  persona = rs.personas['Isabella Rodriguez']
  formatted_thought = persona.format_chat_event(concept_nodes['chat_event'])
  expected_formatted_thought = '''
- At 11:22:40: Isabella Rodriguez is conversing about a conversation about Isabella inviting Klaus to her Valentine's Day party at Hobbs Cafe on February 14th, 2023 from 5pm to 7pm. Here is the dialog history:
  - Isabella Rodriguez: Hi Klaus! How are you enjoying your meal? I wanted to let you know that I'm planning a Valentine's Day party at Hobbs Cafe on February 14th, 2023 from 5pm to 7pm. I would love for you to join us!
  - Klaus Mueller: Oh, hi Isabella! I'm doing well, thank you. The meal is delicious as always. A Valentine's Day party sounds fun. I'd love to join! Thank you for inviting me.
  '''.strip()
  assert formatted_thought == expected_formatted_thought


def test_unit__Persona__format_object_observation_event(rs, concept_nodes):
  persona = rs.personas['Isabella Rodriguez']
  formatted_thought = persona.format_object_observation_event(concept_nodes['object_observation_event'])
  expected_formatted_thought = "- At 00:00:10: In the main room of Isabella Rodriguez's apartment, Isabella Rodriguez is aware that the bed is being used."
  assert formatted_thought == expected_formatted_thought
  log.debug(f'{formatted_thought=}')
