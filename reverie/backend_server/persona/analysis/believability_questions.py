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


def get_chat_interaction_counts(persona):
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

  return chat_counts, dialog_exchange_counts


