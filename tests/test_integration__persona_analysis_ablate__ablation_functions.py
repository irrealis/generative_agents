import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from dotenv import find_dotenv, load_dotenv
env_path = find_dotenv()
load_dotenv(env_path)

import functools as ft
import os
import sys

project_dir = os.path.dirname(os.path.abspath(env_path))
sys.path.insert(0, os.path.abspath(f"{project_dir}/reverie/backend_server"))

import langchain_setup
from reverie_setup import ReverieTestServer, rs

from persona.analysis.ablate import ablate_observations_planning_reflection, ablate_planning_reflection, ablate_reflection, is_planning, is_reflection, is_reflection_error
from persona.analysis.interview import interview_persona


### Tests


def test_integration__ablate_reflection(rs):
  persona = rs.personas['Isabella Rodriguez']
  id_to_node = persona.a_mem.id_to_node
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

  # Sanity checks.
  assert len(persona.a_mem.seq_thought) > 1
  assert len(plans) > 1
  assert len(reflections) > 1
  seq_thought = persona.a_mem.seq_thought[:]
  assert (len(plans) + len(reflections) + len(errors)) == len(seq_thought)

  ablate_reflection(persona)

  # Verify: the thought depth of all remaining thoughts should be 1.
  kw_to_plan_thoughts = {
    kw:[
      node for node in nodes if is_planning(node)
    ]
    for kw,nodes in persona.a_mem.kw_to_thought.items()
  }
  kw_to_thought_depths = {
    kw:set(
      n.depth
      for n in nodes
    )
    for kw,nodes in kw_to_plan_thoughts.items()
  }
  thought_depths = set(ft.reduce(set.union, kw_to_thought_depths.values()))
  assert thought_depths == {1}

  # Verify: the number of planning memories equals the number of remaining
  # thought memories.
  id_to_planning_node = {
    node_id:node
    for node_id, node in persona.a_mem.id_to_node.items()
    if is_planning(node)
  }
  assert len(id_to_planning_node) == len(persona.a_mem.seq_thought)
  # Verify: the depths of all remaining memories should be no more than 1.
  nonreflection_node_depths = set(node.depth for node in id_to_planning_node.values())
  assert max(nonreflection_node_depths) <= 1

  # Verify we've removed thoughts.
  assert len(persona.a_mem.seq_thought) < len(seq_thought)
  assert len(persona.a_mem.id_to_node) < len(id_to_node)

  # Verify consistency between number of remaining thoughts as seen in
  # `seq_thoughts` and `id_to_node`.
  associative_memories = persona.a_mem.id_to_node
  thoughts = {
    key:node
    for key,node in associative_memories.items()
    if node.type == 'thought'
  }
  assert len(thoughts) == len(persona.a_mem.seq_thought)


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
