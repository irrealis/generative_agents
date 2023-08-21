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

from langchain_setup import *
from reverie_setup import ReverieTestServer, rs

import reverie
from reverie import ReverieServer

import pytest

import random
import shutil


random.seed(0)

def test__reverie__Persona__scratch__get_str_iss(rs):
  '''Verify Scratch.get_str_iss() doesn't raise exceptions.'''
  isabella = rs.personas['Isabella Rodriguez']
  # Retrieval of identity stable set
  iss = isabella.scratch.get_str_iss()
  log.debug(
    f'''
*** Persona identity:
{iss}
''')


def test__reverie__Persona__scratch__get_str_daily_schedule_summary(rs):
  '''Verify Scratch.get_str_daily_schedule_summary() doesn't raise exceptions.'''
  isabella = rs.personas['Isabella Rodriguez']
  # Retrieval of daily schedule
  daily_schedule = isabella.scratch.get_str_daily_schedule_summary()
  log.debug(
    f'''
*** Persona daily schedule:
{daily_schedule}
''')


def test__reverie__Persona__scratch__get_str_hourly_schedule_summary(rs):
  '''Verify Scratch.get_str_daily_schedule_hourly_org_summary() doesn't raise exceptions.'''
  isabella = rs.personas['Isabella Rodriguez']
  # Retrieval of hourly schedule
  hourly_schedule = isabella.scratch.get_str_daily_schedule_hourly_org_summary()
  log.debug(
    f'''
*** Persona hourly schedule:
{hourly_schedule}
''')


def test__reverie__Persona__scratch__get_str_seq_events(rs):
  '''Verify AssociativeMemory.get_str_seq_events() doesn't raise exceptions.'''
  isabella = rs.personas['Isabella Rodriguez']
  # Retrieval of associative memory (event)
  event_memories = isabella.a_mem.get_str_seq_events()
  log.debug(
    f'''
*** Persona event memories:
{event_memories}
''')


def test__reverie__Persona__scratch__get_str_seq_thoughts(rs):
  '''Verify AssociativeMemory.get_str_seq_thoughts() doesn't raise exceptions.'''
  isabella = rs.personas['Isabella Rodriguez']
  # Retrieval of associative memory (thought)
  thought_memories = isabella.a_mem.get_str_seq_thoughts()
  log.debug(
    f'''
*** Persona thought memories:
{thought_memories}
''')


def test__reverie__Persona__scratch__get_str_seq_chats(rs):
  '''Verify AssociativeMemory.get_str_seq_chats() doesn't raise exceptions.'''
  isabella = rs.personas['Isabella Rodriguez']
  # Rewrite of method body that correctly accesses 
  chat_memories = isabella.a_mem.get_str_seq_chats()
  log.debug(
    f'''
*** Persona chat memories:
{chat_memories}
''')
