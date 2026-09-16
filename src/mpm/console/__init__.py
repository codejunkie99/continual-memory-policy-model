"""Read-only local Cognition Console on top of the MPM runtime.

Public surface:

* :class:`~mpm.console.views.ConsoleView` - read-only projection of a memory
  store into JSON-serializable dictionaries.
* :func:`~mpm.console.server.make_server` - build a localhost HTTP server
  without starting it.
* :func:`~mpm.console.server.is_loopback` - report whether a bind host is a
  loopback address.
"""

from __future__ import annotations

from .server import is_loopback, make_server
from .views import ConsoleView

__all__ = ["ConsoleView", "make_server", "is_loopback"]
