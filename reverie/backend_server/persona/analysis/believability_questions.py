"""
Author: Irrealis (irrealis-chomp@gmail.com)

File: believability_questions.py
Description: These functions are for setting up persona interview questions to
analyze believability of persona behavior under various ablation conditions
described in Park's paper "Generative Agents: Interactive Simulacra of Human
Behavior."
"""

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import random


def get_chat_interaction_counts(persona):
  '''
  Counts persona's chat interactions with other personas.

  Returns two counts for each persona with whom a subject persona has chatted:

  - Number of chats with the other persona
  - Total number dialog exchanges with the other persona.
  '''
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
        # Increment count of dialog exchanges with chat participant.
        dialog_exchange_count = dialog_exchange_counts.get(speaker, 0)
        dialog_exchange_count += 1
        dialog_exchange_counts[speaker] = dialog_exchange_count

  return chat_counts, dialog_exchange_counts


def get_max_chat_interactions(persona):
  chat_counts, dialog_exchange_counts = get_chat_interaction_counts(persona)
  # Note: there can be ties in these maxima. Python will choose one, but I'm not sure how.
  max_chats = max(chat_counts.items(), key=lambda x: x[1])
  max_dialog_exchanges = max(dialog_exchange_counts.items(), key=lambda x: x[1])
  return max_chats, max_dialog_exchanges


def get_believability_question_variables(
  persona,
  personas,
  random_persona_clause = None,
  event = None,
  random_seed = None
):
  if random_persona_clause is None:
    random_persona_clause = "organizing a Valentine's Day party"

  if event is None:
    event = "a Valentine's Day party"

  if random_seed is None:
    rng = random.Random()
  else:
    rng = random.Random(random_seed)

  persona_names = list(personas.keys())
  persona_names.remove(persona.name)

  random_persona_name = rng.choice(persona_names)
  random_persona_1 = personas[random_persona_name]
  persona_names.remove(random_persona_1.name)
  random_persona_name = rng.choice(persona_names)
  random_persona_2 = personas[random_persona_name]
  persona_names.remove(random_persona_2.name)

  max_chats, max_dialog_exchanges = get_max_chat_interactions(persona)
  well_known_persona_name = max_dialog_exchanges[0]

  question_variables = dict(
    random_persona_name_1 = random_persona_1.name,
    random_persona_name_2 = random_persona_2.name,
    random_persona_clause = random_persona_clause,
    event = event,
    well_known_persona_name = well_known_persona_name,
  )

  return question_variables
