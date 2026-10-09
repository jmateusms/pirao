# Releasing to PyPI

The Python package lives in `backend/`; the web app is built into it, so a
release needs Node once, on the maintainer's machine, and never on a user's.

1. Bump the version in `backend/pyproject.toml`, `backend/src/pirao/__init__.py`,
   `frontend/package.json` (and its lock file, with
   `npm version X.Y.Z --no-git-tag-version` in `frontend/`) and `CITATION.cff`
   (`version`, `date-released`); add an entry to
   `CHANGELOG.md`. Merge that to `main`.
2. Build the interface into the package, then the distributions:

   ```bash
   cd frontend && npm ci && npm run build:package && cd ..
   cd backend && rm -rf dist && uv build
   ```

   `build:package` writes `backend/src/pirao/gui_static/` (not in git); both the
   wheel and the sdist include it. Check it is there:
   `unzip -l dist/*.whl | grep gui_static/index.html`.
3. Try the wheel in a clean environment: `pirao install-stan`, `pirao examples`,
   `pirao gui`.
4. Upload and tag:

   ```bash
   uvx twine check dist/*
   uvx twine upload dist/*
   git tag -a vX.Y.Z -m "pirão X.Y.Z" && git push origin vX.Y.Z
   ```
