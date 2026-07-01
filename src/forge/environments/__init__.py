"""Reserved namespace — concrete environments are NOT framework code.

Concrete environments now live in the repo-root ``envs/`` plugin tree and register via an
experiment's ``plugins:`` field (see ``forge.core.plugins``). This package is kept only as a
reserved namespace; ``src/forge/`` owns protocols/ABCs/generic utilities, never concrete env code.
"""

