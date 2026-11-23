# Deploying CanopyAI to Streamlit Community Cloud

The app has two modes, and they need different things on the server:

| Mode | Needs Earth Engine on the server? |
|---|---|
| Preloaded cities (São Paulo, Rio, Madrid, Phoenix) | **No.** Grids and default plans ship in `data/demo/` |
| Custom area (draw a rectangle → download from Earth Engine) | **Yes.** A Google Cloud *service account* in `st.secrets` |

A deployment with no secrets at all is valid: the preloaded cities work, and the custom-area button
explains that downloads are disabled.

---

## 1. Push the repository to GitHub

```bash
# check that no secret is tracked before the first push
git ls-files | grep -iE "\.env$|secrets\.toml$|\.json$"
# expected output: only reports/cross_model_audit.json

git remote add origin https://github.com/<your-user>/canopyai.git
git push -u origin main
```

`.env`, `.streamlit/secrets.toml` and `data/cache/` are git-ignored. **Keep the service-account key
file outside the repository folder.** Its default file name (`<project>-<hash>.json`) does not match
any ignore rule.

## 2. Create the app on Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> with GitHub and click **Create app → Deploy a public app
   from GitHub**.
2. Repository `<your-user>/canopyai`, branch `main`, main file `app.py`.
3. **Advanced settings → Python version: 3.12** (or 3.13 / 3.14). Every pinned package in
   `requirements.txt` has Linux wheels for these versions; see *Verification* below.
4. Leave *Secrets* empty for a demo-only deployment, or follow §3 first. Click **Deploy**.

The first build installs `requirements.txt` (a few minutes). Streamlit Cloud reads the theme from
`.streamlit/config.toml`.

## 3. Enable custom areas (Earth Engine service account)

The Cloud project must already be **registered for Earth Engine** (noncommercial or commercial) with
the Earth Engine API enabled. This is the same project you use locally (`GEE_PROJECT_ID`).

### 3.1 Create the service account and key

In the Google Cloud console, with that project selected:

1. **IAM & Admin → Service Accounts → Create service account**, for example `canopyai-app`.
2. Grant it the roles Earth Engine documents for service accounts that run computations:
   - **Earth Engine Resource Viewer** (`roles/earthengine.viewer`). The app only reads public
     datasets and computes pixels; it never writes assets.
   - **Service Usage Consumer** (`roles/serviceusage.serviceUsageConsumer`).

   If the first download fails with a permission error, compare these roles with the current
   [Earth Engine service-account guide](https://developers.google.com/earth-engine/guides/service_account).
3. Open the account → **Keys → Add key → Create new key → JSON**. The file downloads once. Store it
   outside the repository.

### 3.2 Put the key in Streamlit secrets

Convert the JSON key into the TOML block that Streamlit expects:

```bash
python scripts/key_to_toml.py "C:\path\to\key.json"
```

In the Streamlit Cloud app, open **⋮ → Settings → Secrets** and paste:

```toml
GEE_PROJECT_ID = "your-gcp-project-id"
CANOPYAI_ALLOW_CUSTOM = "1"     # "0" turns new downloads off without redeploying
CANOPYAI_MAX_DOWNLOADS = "3"    # new areas per browser session

[gcp_service_account]
# ... the lines printed by key_to_toml.py ...
```

`.streamlit/secrets.toml.example` has the full shape. When the app starts it copies these values
into the environment variables the Earth Engine code reads (`GEE_SERVICE_ACCOUNT_JSON`,
`GEE_PROJECT_ID`, …) and initializes Earth Engine with `ee.ServiceAccountCredentials`. The browser
login flow is disabled in the app (`CANOPYAI_HEADLESS=1`), so a missing or invalid key shows an error
message instead of hanging the server.

### 3.3 Test locally with the same secrets (optional)

Save the same content as `.streamlit/secrets.toml` (git-ignored) and run `streamlit run app.py`.
With secrets present, the service account takes precedence over your personal
`earthengine authenticate` credentials.

## 4. Protecting your Earth Engine quota

A public app runs every custom-area download under **your** service account. The guards in place:

- **Area cap:** rectangles above **100 km²** are rejected before any request
  (`MAX_CUSTOM_AREA_KM2` in `app_core/cities.py`). A 100 km² area is ~110,000 cells, served by one or
  two 512 × 512-pixel `computePixels` requests in about 5–20 s.
- **Per-session cap:** `CANOPYAI_MAX_DOWNLOADS` new areas per browser session. This deters
  accidental overuse, not a determined user, because a new session resets it.
- **Kill switch:** set `CANOPYAI_ALLOW_CUSTOM = "0"` in Secrets to serve the demo cities only.
- **Cache:** an area that was already downloaded is served from disk with no Earth Engine call.

Also watch usage in the Cloud console (*Earth Engine → Usage* and *APIs & Services → Quotas*). If the
project has billing, set a budget alert.

## 5. Things to know about the hosted environment

- **Storage is ephemeral.** Areas downloaded on the server live in `data/cache/` until the app
  restarts or sleeps. Only `data/demo/` is permanent, because it is in git.
- **New scenarios are compute-bound.** A scenario whose plan is not cached (for example a different
  tree budget) runs the lazy greedy on the shared CPU: about 1–2 minutes locally, likely longer in the
  cloud. Default scenarios open instantly.
- **Updating the demo data:** run `python scripts/precache_cities.py --export-demo` locally, then commit
  `data/demo/` and push. Changing `CACHE_SCHEMA_VERSION` (`gee/extract.py`) or `PLAN_VERSION`
  (`app_core/plan.py`) invalidates old files, so re-export after either change.

## Verification

Before writing this guide, the repository was checked as follows:

- `pip install --dry-run --report … --platform manylinux_2_28_x86_64 --only-binary=:all:` resolved
  the full dependency tree (64 packages) to Linux wheels for Python 3.12, 3.13 and 3.14. There are no
  source builds and no Windows-only packages.
- The app was run with `data/cache/` removed, so only `data/demo/` was available. All four cities
  and both objectives loaded without errors (Streamlit `AppTest`).
- `scripts/key_to_toml.py` was round-tripped on a dummy key. The TOML parsed back to the identical
  JSON, including the private key's newlines.
