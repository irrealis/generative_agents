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
from persona.analysis.interview import interview_persona


### Tests


def test_integration__ablate_planning_reflection(rs):
  persona = rs.personas['Isabella Rodriguez']

  # Sanity checks.
  assert len(persona.a_mem.seq_thought) > 1

  ablate_planning_reflection(persona)

  assert len(persona.a_mem.id_to_node) > 1
  assert len(persona.a_mem.seq_event) > 1
  assert len(persona.a_mem.seq_chat) > 0
  assert len(persona.a_mem.seq_thought) == 0
  assert len(persona.a_mem.kw_to_chat) > 0
  assert len(persona.a_mem.kw_to_thought) == 0
  assert len(persona.scratch.chatting_with_buffer) > 0
  assert len(persona.scratch.daily_req) == 0
  assert len(persona.scratch.daily_plan_req) == 0
  assert len(persona.scratch.f_daily_schedule) == 0
  assert len(persona.scratch.f_daily_schedule_hourly_org) == 0


# This brainstorm prototypes a procedure to interview a persona under partial
# ablation: no planning or reflection memories.
#
# This is simpler than the full ablation because:
# - Removing all plans and reflections equates to removing all thought memories.
# - Because non-thought memories are not removed, there are still enough
#   memories present that no special effort is required to avoid the kinds of
#   problems encountered in full ablation.
#
def test_brainstorm__prototype__persona_ablations__planning_reflections(rs):
  persona = rs.personas['Isabella Rodriguez']

  # Wipe all thought memories.
  persona.a_mem.seq_thought = []
  persona.a_mem.kw_to_thought = dict()
  persona.a_mem.id_to_node = {
    node_id:node
    for node_id, node in persona.a_mem.id_to_node.items()
    if node.type != 'thought'
  }
  persona.scratch.daily_req = []
  persona.scratch.daily_plan_req = []
  persona.scratch.f_daily_schedule = []
  persona.scratch.f_daily_schedule_hourly_org = []

  # Interview agent with partial ablation.
  question = 'Give an introduction of yourself.'
  response, current_convo = interview_persona(
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
