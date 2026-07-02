"""Repo-root pytest config.

Defensively excludes the two standalone repo-root scripts from collection.
They execute their whole suite at import time and call sys.exit(), which makes
pytest INTERNALERROR to "0 collected". testpaths=tests already scopes bare
`pytest` to the suite; this also covers `pytest .` / rootdir discovery.
Run those tools directly: `python test_tar_comprehensive.py`.
"""

collect_ignore = [
    "test_tar_comprehensive.py",
    "test_tar_integration.py",
]
