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

import langchain_setup
from reverie_setup import ReverieTestServer, rs

from persona.analysis.ablate import ablate_observations_planning_reflection, ablate_planning_reflection, ablate_reflection, is_planning, is_reflection, is_reflection_error


### Tests


def test_integration__ablate_observations_planning_reflection(rs):
  persona = rs.personas['Isabella Rodriguez']

  # Sanity checks.
  assert len(persona.a_mem.seq_event) > 1
  assert len(persona.a_mem.seq_chat) > 1
  assert len(persona.a_mem.seq_thought) > 1

  ablate_observations_planning_reflection(persona)

  assert len(persona.a_mem.id_to_node) == 1
  assert len(persona.a_mem.seq_event) == 1
  assert len(persona.a_mem.seq_chat) == 0
  assert len(persona.a_mem.seq_thought) == 0
  assert len(persona.a_mem.kw_to_chat) == 0
  assert len(persona.a_mem.kw_to_thought) == 0
  assert len(persona.scratch.chatting_with_buffer) == 0
  assert len(persona.scratch.daily_req) == 0
  assert len(persona.scratch.daily_plan_req) == 0
  assert len(persona.scratch.f_daily_schedule) == 0
  assert len(persona.scratch.f_daily_schedule_hourly_org) == 0
