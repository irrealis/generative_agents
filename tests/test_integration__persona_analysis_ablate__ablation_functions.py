import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


### Tests


def test_hello():
  log.debug('Hello, world.')
