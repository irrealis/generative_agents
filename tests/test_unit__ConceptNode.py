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

### Tests


def test_unit__ConceptNode__is_plan_thought(concept_nodes):
  assert not concept_nodes['chat_event'].is_plan_thought()
  assert not concept_nodes['object_observation_event'].is_plan_thought()
  assert not concept_nodes['activity_event'].is_plan_thought()
  assert concept_nodes['plan_thought'].is_plan_thought()
  assert not concept_nodes['reflection_thought'].is_plan_thought()
  assert not concept_nodes['reflection_error_thought'].is_plan_thought()
  assert not concept_nodes['bootstrap_thought'].is_plan_thought()
  assert not concept_nodes['chat'].is_plan_thought()


def test_unit__ConceptNode__is_reflection_thought(concept_nodes):
  assert not concept_nodes['chat_event'].is_reflection_thought()
  assert not concept_nodes['object_observation_event'].is_reflection_thought()
  assert not concept_nodes['activity_event'].is_reflection_thought()
  assert not concept_nodes['plan_thought'].is_reflection_thought()
  assert concept_nodes['reflection_thought'].is_reflection_thought()
  assert not concept_nodes['reflection_error_thought'].is_reflection_thought()
  assert not concept_nodes['bootstrap_thought'].is_reflection_thought()
  assert not concept_nodes['chat'].is_reflection_thought()


def test_unit__ConceptNode__is_reflection_error_thought(concept_nodes):
  assert not concept_nodes['chat_event'].is_reflection_error_thought()
  assert not concept_nodes['object_observation_event'].is_reflection_error_thought()
  assert not concept_nodes['activity_event'].is_reflection_error_thought()
  assert not concept_nodes['plan_thought'].is_reflection_error_thought()
  assert not concept_nodes['reflection_thought'].is_reflection_error_thought()
  assert concept_nodes['reflection_error_thought'].is_reflection_error_thought()
  assert not concept_nodes['bootstrap_thought'].is_reflection_error_thought()
  assert not concept_nodes['chat'].is_reflection_error_thought()


def test_unit__ConceptNode__is_bootstrap_thought(concept_nodes):
  assert not concept_nodes['chat_event'].is_bootstrap_thought()
  assert not concept_nodes['object_observation_event'].is_bootstrap_thought()
  assert not concept_nodes['activity_event'].is_bootstrap_thought()
  assert not concept_nodes['plan_thought'].is_bootstrap_thought()
  assert not concept_nodes['reflection_thought'].is_bootstrap_thought()
  assert not concept_nodes['reflection_error_thought'].is_bootstrap_thought()
  assert concept_nodes['bootstrap_thought'].is_bootstrap_thought()
  assert not concept_nodes['chat'].is_bootstrap_thought()


def test_unit__ConceptNode__is_chat_event(concept_nodes):
  assert concept_nodes['chat_event'].is_chat_event()
  assert not concept_nodes['object_observation_event'].is_chat_event()
  assert not concept_nodes['activity_event'].is_chat_event()
  assert not concept_nodes['plan_thought'].is_chat_event()
  assert not concept_nodes['reflection_thought'].is_chat_event()
  assert not concept_nodes['reflection_error_thought'].is_chat_event()
  assert not concept_nodes['bootstrap_thought'].is_chat_event()
  assert not concept_nodes['chat'].is_chat_event()


def test_unit__ConceptNode__is_object_observation_event(concept_nodes):
  assert not concept_nodes['chat_event'].is_object_observation_event()
  assert concept_nodes['object_observation_event'].is_object_observation_event()
  assert not concept_nodes['activity_event'].is_object_observation_event()
  assert not concept_nodes['plan_thought'].is_object_observation_event()
  assert not concept_nodes['reflection_thought'].is_object_observation_event()
  assert not concept_nodes['reflection_error_thought'].is_object_observation_event()
  assert not concept_nodes['bootstrap_thought'].is_object_observation_event()
  assert not concept_nodes['chat'].is_object_observation_event()


def test_unit__ConceptNode__is_activity_event(concept_nodes):
  assert not concept_nodes['chat_event'].is_activity_event()
  assert not concept_nodes['object_observation_event'].is_activity_event()
  assert concept_nodes['activity_event'].is_activity_event()
  assert not concept_nodes['plan_thought'].is_activity_event()
  assert not concept_nodes['reflection_thought'].is_activity_event()
  assert not concept_nodes['reflection_error_thought'].is_activity_event()
  assert not concept_nodes['bootstrap_thought'].is_activity_event()
  assert not concept_nodes['chat'].is_activity_event()


def test_unit__ConceptNode__is_chat(concept_nodes):
  assert not concept_nodes['chat_event'].is_chat()
  assert not concept_nodes['object_observation_event'].is_chat()
  assert not concept_nodes['activity_event'].is_chat()
  assert not concept_nodes['plan_thought'].is_chat()
  assert not concept_nodes['reflection_thought'].is_chat()
  assert not concept_nodes['reflection_error_thought'].is_chat()
  assert not concept_nodes['bootstrap_thought'].is_chat()
  assert concept_nodes['chat'].is_chat()


def test_unit__ConceptNode__is_idle(concept_nodes):
  assert not concept_nodes['chat_event'].is_idle()
  assert not concept_nodes['object_observation_event'].is_idle()
  assert not concept_nodes['activity_event'].is_idle()
  assert not concept_nodes['plan_thought'].is_idle()
  assert not concept_nodes['reflection_thought'].is_idle()
  assert not concept_nodes['reflection_error_thought'].is_idle()
  assert not concept_nodes['bootstrap_thought'].is_idle()
  assert not concept_nodes['chat'].is_idle()
  assert concept_nodes['idle_activity_event'].is_idle()
  assert concept_nodes['idle_object_observation_event'].is_idle()


def test_prototype__ConceptNode__classify(concept_nodes):
  assert ['chat_event'] == concept_nodes['chat_event'].classify()
  assert ['object_observation_event'] == concept_nodes['object_observation_event'].classify()
  assert ['activity_event'] == concept_nodes['activity_event'].classify()
  assert ['plan_thought'] == concept_nodes['plan_thought'].classify()
  assert ['reflection_thought'] == concept_nodes['reflection_thought'].classify()
  assert ['reflection_error_thought'] == concept_nodes['reflection_error_thought'].classify()
  assert ['bootstrap_thought'] == concept_nodes['bootstrap_thought'].classify()
  assert ['chat'] == concept_nodes['chat'].classify()
