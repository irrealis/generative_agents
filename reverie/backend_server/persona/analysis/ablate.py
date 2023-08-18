"""
Author: Irrealis (irrealis-chomp@gmail.com)

File: ablate.py
Description: Code to perform ablations as described in Park et al. paper "Generative
Agents: Interactive Simulacra of Human Behavior."
"""

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from ..cognitive_modules.perceive import generate_poig_score
from ..prompt_template.gpt_structure import get_embedding


def is_planning(node):
  return (node.type == 'thought') and (node.filling is None)

def is_reflection(node):
  return (node.type == 'thought') and isinstance(node.filling, list)

def is_reflection_error(node):
  return (node.type == 'thought') and isinstance(node.filling, str)


def ablate_reflection(persona):
  # Remove non-planning thoughts from keyword-nodes mapping.
  kw_to_plan_thoughts = {
    kw:[
      node for node in nodes if is_planning(node)
    ]
    for kw,nodes in persona.a_mem.kw_to_thought.items()
  }
  # The above may result in keywords with empty memory lists.
  # Below such keywords are removed.
  kw_to_plan_thoughts = {kw:nodes for kw,nodes in kw_to_plan_thoughts.items() if nodes}

  # Remove nodes that are either reflections or reflection-errors from id-nodes mapping.
  id_to_nonreflection_node = {
    node_id:node
    for node_id, node in persona.a_mem.id_to_node.items()
    if not (is_reflection(node) or is_reflection_error(node))
  }

  # Wipe non-plan thoughts.
  seq_planning_thought = [node for node in persona.a_mem.seq_thought if is_planning(node)]

  # Update persona's associative memory.
  persona.a_mem.kw_to_thought = kw_to_plan_thoughts
  persona.a_mem.id_to_node = id_to_nonreflection_node
  persona.a_mem.seq_thought = seq_planning_thought


def ablate_planning_reflection(persona):
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


def ablate_observations_planning_reflection(persona):
  initial_event_memory = persona.a_mem.seq_event[-1]
  s = persona.name
  p = 'is'
  o = persona.name
  event_description = f'{s} {p} {o}'
  event_keywords = {s}
  event_embedding = get_embedding(event_description)
  event_embedding_pair = (event_description, event_embedding)
  event_poignancy = generate_poig_score(persona, "event", event_description)
  persona.a_mem.add_event(
    created = initial_event_memory.created,
    expiration = None,
    s = s,
    p = p,
    o = o,
    description = event_description,
    keywords = event_keywords,
    poignancy = event_poignancy,
    embedding_pair = event_embedding_pair,
    filling = [],
  )

  # Extract the empty memory.
  empty_memory = persona.a_mem.seq_event[0]

  # Give it bogus ID information.
  empty_memory.node_id = 'node_0'
  empty_memory.node_count = 0
  empty_memory.type_count = 0
  persona.a_mem.seq_event[0] = empty_memory
  persona.a_mem.seq_event = [empty_memory]

  # Wipe all other memories.
  narrowed_kw_to_event = {
    k:[node for node in nodes if node.node_id == empty_memory.node_id]
    for k,nodes in persona.a_mem.kw_to_event.items()
    if empty_memory.node_id in [
      node.node_id
      for node in nodes
    ]
  }
  persona.a_mem.kw_to_event = narrowed_kw_to_event
  persona.a_mem.id_to_node = {
    empty_memory.node_id: empty_memory
  }
  persona.scratch.chatting_with = None
  persona.scratch.chat = None
  persona.scratch.chatting_with_buffer = []
  persona.scratch.daily_req = []
  persona.scratch.daily_plan_req = []
  persona.scratch.f_daily_schedule = []
  persona.scratch.f_daily_schedule_hourly_org = []
  persona.a_mem.seq_thought = []
  persona.a_mem.seq_chat = []

  persona.a_mem.kw_to_thought = dict()
  persona.a_mem.kw_to_chat = dict()
