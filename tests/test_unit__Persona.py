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
