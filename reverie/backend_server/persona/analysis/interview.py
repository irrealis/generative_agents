"""
Author: Irrealis (irrealis-chomp@gmail.com)

File: interview.py
Description: This provides code for evaluation of persona behaviors via
interviews under the ablation conditions described in Park's paper "Generative
Agents: Interactive Simulacra of Human Behavior."
"""

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from ..cognitive_modules.retrieve import new_retrieve
from ..cognitive_modules.converse import generate_next_line,  generate_summarize_ideas


def interview_persona(persona, message, curr_convo=None, interviewer=None, n_count=None):
  if curr_convo is None:
    curr_convo = []
  if interviewer is None:
    interviewer = 'Interviewer'

  retrieved = new_retrieve(
    persona=persona,
    focal_points=[message],
    n_count=n_count,
    weights=(1.,1.,1.),
  )[message]
  summarized_idea = generate_summarize_ideas(persona, retrieved, message)
  curr_convo += [[interviewer, message]]
  response = generate_next_line(persona, interviewer, curr_convo, summarized_idea)
  curr_convo += [[persona.scratch.name, response]]

  return response, curr_convo

