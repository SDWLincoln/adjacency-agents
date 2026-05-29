"""Provider adapters.

Adapters live in their own subpackage so that the core library has no
hard dependency on any specific LLM SDK. Each adapter module imports
its provider SDK lazily when needed, and accepts the SDK client object
directly so that tests can substitute lightweight fakes.
"""
