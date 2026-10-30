"""Importing this package registers every built-in component.

Optional third-party backends are imported lazily *inside* each class, so this
import stays cheap and never fails because a provider SDK is missing.
"""

from app.rag.components import (  # noqa: F401
    chunkers,
    compressors,
    embedders,
    generators,
    llms,
    loaders,
    rerankers,
    retrievers,
    transforms,
    vectorstores,
)
