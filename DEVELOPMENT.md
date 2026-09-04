# Development

## Publish an internal release candidate

The package version is a PEP 440 release candidate such as `0.1.0rc1`. Put the internal
Artifactory upload endpoint and your access or identity token in the gitignored `.env`:

```dotenv
ARTIFACTORY_PYPI_URL=<internal-artifactory-pypi-upload-url>
ARTIFACTORY_TOKEN=<token>
```

Use the full token from Artifactory. The accompanying reference token is a shorter alias and is
not needed by this publisher.

Build and validate without uploading:

```bash
uv run tools/publish_artifactory.py --dry-run
```

Publish the wheel to internal Artifactory:

```bash
uv run tools/publish_artifactory.py
```

The script builds a clean wheel, rejects environment files, refuses non-NVIDIA upload endpoints,
disables trusted/public publishing, uploads with `uv publish`, submits SHA-1 and SHA-256 client
checksums, downloads the resulting artifact, and verifies its SHA-256 checksum. It refuses to
overwrite an existing version with different bytes and then prints the direct wheel URL to register
with nSpect. It refuses non-RC versions.

Both CLI forms are supported after installing the wheel:

```bash
insight-agent --version
python -m insight_agent --version
```
