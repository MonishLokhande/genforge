"""genforge bundled environments — concrete, swappable data-source plugins.

Each ``envs/<env>/`` package is a self-contained plugin: an ``Environment`` (raw data source), a
``Dataset`` (a :class:`forge.core.protocols.BaseDataset`), and a ``Processor`` (a
:class:`forge.core.protocols.BaseProcessor`). Packages register their components via
``@register`` when imported, and are loaded through an experiment's ``plugins:`` field — never by
``genforge``'s built-in module list. ``envs.common`` holds the env-agnostic ``distribution`` dataset
shared across the sampling families. See ``forge.core.plugins`` for the loader.
"""
