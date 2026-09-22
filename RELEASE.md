# Releasing

Publishing uses **PyPI Trusted Publishing**: PyPI authenticates this repository's GitHub
Actions OIDC identity directly, so there is no API token to create, store or rotate, and
every artifact carries a [PEP 740](https://peps.python.org/pep-0740/) attestation binding it
to the commit and workflow that built it.

## One-time setup: registering the project

The project does not have to exist on PyPI first. Adding a **pending publisher** both
reserves the name and creates the project on the first successful publish, so there is
nothing to upload by hand and no token anywhere.

1. Enable 2FA on the PyPI account if it is not already on — trusted publishing requires it.
2. Check that the name is free: <https://pypi.org/project/onnx-ad/> should be a 404. PyPI
   normalizes names, so `onnx-ad`, `onnx_ad` and `onnx.ad` are all the same project.
3. Go to <https://pypi.org/manage/account/publishing/>, scroll to
   **Add a new pending publisher**, choose **GitHub**, and fill in exactly:

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `onnx-ad` |
   | Owner | `yacoda` |
   | Repository name | `onnx-ad` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The workflow name is the **file name**, not a path and not the `name:` inside the file.
   The environment name must match the `environment:` of the publishing job exactly.
4. In the GitHub repository, **Settings → Environments → New environment**, named `pypi`.
   Add a required reviewer (yourself) and, under deployment branches, restrict it to the tag
   pattern `v*`. Nothing then publishes without a deliberate approval from a real tag.

That is the whole registration. The pending publisher becomes an ordinary trusted publisher
attached to the project once the first release succeeds.

## Each release

1. Update `version` in `pyproject.toml` and `__version__` in `src/onnx_ad/__init__.py` —
   the two must agree — and move the `CHANGELOG.md` entry out of "unreleased".
2. Dry run: **Actions → Release → Run workflow**. On a manual run the workflow builds,
   runs `twine check --strict`, and stops; download the `distributions` artifact and
   `pip install` the wheel in a clean environment to confirm it imports and the console
   script works. Nothing is published.
3. Tag and publish a GitHub release named `v<version>`. That triggers the `pypi` job, which
   waits for the environment approval and then uploads. The build job refuses to continue if
   the tag and the version in `pyproject.toml` disagree.

## Verifying provenance

The project page on PyPI shows a **Provenance** panel for every file published this way,
naming the repository, the workflow file and the commit the artifact was built from. A wheel
built anywhere else cannot carry one.

For a check in CI or on the command line, the `pypi-attestations` tool verifies a downloaded
distribution against its attestation; see its `--help` for the current invocation.
