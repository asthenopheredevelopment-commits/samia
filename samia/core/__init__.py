"""samia.core — pure library plane.

Modules are extracted from the existing memory/tools/*.py scripts. All public
APIs are pure functions or dataclasses; no daemon required.

Acceptance (design doc §8.1): existing scripts refactored to import from
samia.core must produce byte-identical output to the pre-refactor versions.
"""
