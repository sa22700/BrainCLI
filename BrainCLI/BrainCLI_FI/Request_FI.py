"""
Copyright [2025] [Pirkka Toivakka]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# This project uses model weights licensed under CC BY 4.0 (see /Models/LICENSE)


from __future__ import annotations
import json
import os
import ssl
import socket
import shutil
import textwrap
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple, Optional
from urllib.error import HTTPError, URLError
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

DEFAULT_BASE_URL = os.environ.get("BRAINCLI_SEARCH_URL", "").strip()
MULTI_BASE_URLS = [u.strip() for u in os.environ.get("BRAINCLI_SEARCH_URLS", "").split(",") if u.strip()]
DEFAULT_LIMIT    = int(os.environ.get("BRAINCLI_SEARCH_LIMIT", "5"))
DEFAULT_LANG     = os.environ.get("BRAINCLI_SEARCH_LANG", "fi")
DEFAULT_SAFE     = os.environ.get("BRAINCLI_SAFE", "moderate")
DEFAULT_TIME     = os.environ.get("BRAINCLI_TIME", "")
DEFAULT_ENGINES  = os.environ.get("BRAINCLI_ENGINES", "")
DEFAULT_CATS     = os.environ.get("BRAINCLI_CATEGORIES", "")
ALLOW_HTTP       = os.environ.get("BRAINCLI_ALLOW_HTTP", "0") == "1"
CA_FILE          = os.environ.get("BRAINCLI_CA_FILE", "").strip()
REQUEST_TIMEOUT  = float(os.environ.get("BRAINCLI_SEARCH_TIMEOUT", "15"))
SHOW_SNIPPETS    = os.environ.get("BRAINCLI_SHOW_SNIPPETS", "1") == "1"
SNIPPET_LEN      = int(os.environ.get("BRAINCLI_SNIPPET_LEN", "240"))
_TERM_COLS       = shutil.get_terminal_size((100, 24)).columns
WRAP_COLS        = int(os.environ.get("BRAINCLI_WRAP", str(min(100, _TERM_COLS))))

def _safesearch_to_int(val: str | None) -> int:
    if not val:
        return 1
    v = val.lower()
    if v in ("0", "off", "none"):
        return 0
    if v in ("2", "strict", "max"):
        return 2
    return 1

def _ensure_search_path(base_url: str) -> str:
    p = urllib.parse.urlparse(base_url)
    path = p.path or ""
    if path.rstrip("/") != "/search":
        if path.endswith("/"):
            path = path + "search"
        elif path:
            path = path + "/search"
        else:
            path = "/search"
    else:
        path = "/search"
    return urllib.parse.urlunparse(p._replace(path=path))


def _make_url(base_url: str, params: Dict[str, str]) -> str:
    base = _ensure_search_path(base_url)
    q = urllib.parse.urlencode(params, doseq=True, safe=":")
    p = urllib.parse.urlparse(base)
    return urllib.parse.urlunparse(p._replace(query=q))

def _pretty_lines(items: List[dict], limit: int) -> str:
    if not items:
        return "Ei tuloksia."

    def _clean(s: str) -> str:
        return " ".join((s or "").replace("\r", " ").replace("\n", " ").split())

    out_lines: List[str] = []
    maxn = max(1, limit)
    for r in items[:maxn]:
        title   = _clean(r.get("title") or "") or "(ei otsikkoa)"
        link    = (r.get("url") or "").strip()
        date    = (r.get("publishedDate") or r.get("pubdate") or "").strip()
        out_lines.append(f"- {title}")
        meta = f"  {link}"
        if date:
            meta += f"  ({date[:10]})"
        out_lines.append(meta)
        if SHOW_SNIPPETS:
            snippet = _clean(r.get("content") or r.get("snippet") or r.get("abstract") or "")
            if snippet:
                if 0 < SNIPPET_LEN < len(snippet):
                    snippet = snippet[:SNIPPET_LEN - 1].rstrip() + "…"
                out_lines.append(textwrap.fill(
                    snippet,
                    width=max(40, WRAP_COLS),
                    initial_indent="  ",
                    subsequent_indent="  ",
                    replace_whitespace=False,
                    drop_whitespace=True,
                ))
    return "\n".join(out_lines)

def _build_ssl_context() -> Optional[ssl.SSLContext]:
    ctx = ssl.create_default_context()
    if CA_FILE:
        try:
            ctx.load_verify_locations(cafile=CA_FILE)
        except Exception as e:
            log_error(f"CA-tiedoston lataus epäonnistui: {CA_FILE} ({e})")
            raise RuntimeError(f"CA-tiedoston lataus epäonnistui: {CA_FILE} ({e})")
    return ctx

def _candidate_bases() -> List[str]:
    seen = set()
    out: List[str] = []
    for u in [DEFAULT_BASE_URL, *MULTI_BASE_URLS]:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out

def fetch_url(args: str = "") -> str:
    try:
        query = (args or "").strip()
        if not query:
            return "Hae verkosta."
        params = {"q": query, "format": "json"}
        if DEFAULT_LANG:    params["language"]    = DEFAULT_LANG
        if DEFAULT_TIME:    params["time_range"]  = DEFAULT_TIME
        if DEFAULT_ENGINES: params["engines"]     = DEFAULT_ENGINES
        if DEFAULT_CATS:    params["categories"]  = DEFAULT_CATS
        params["safesearch"] = str(_safesearch_to_int(DEFAULT_SAFE))
        bases = _candidate_bases()
        if not bases:
            return "Hakupalvelimen osoite puuttuu."
        last_err: Optional[Tuple[str, Exception]] = None
        for base in bases:
            try:
                url = _make_url(base, params)
                scheme = urllib.parse.urlparse(url).scheme
                if scheme != "https" and not ALLOW_HTTP:
                    return "Vain HTTPS on sallittu"
                headers = {
                    "User-Agent": "BrainCLI/1.1 (+local)",
                    "Accept": "application/json",
                }
                req = urllib.request.Request(url, headers=headers)
                context = _build_ssl_context() if scheme == "https" else None
                with urllib.request.urlopen(req, context=context, timeout=REQUEST_TIMEOUT) as resp:
                    charset = resp.headers.get_content_charset() or "utf-8"
                    raw = resp.read().decode(charset, errors="replace")
                data = json.loads(raw)
                items = data.get("results", [])
                return _pretty_lines(items, DEFAULT_LIMIT)

            except HTTPError as e:
                log_error(f"HTTPError: {e}")
                last_err = (base, e)
                continue
            except (URLError, socket.timeout, ssl.SSLError, json.JSONDecodeError) as e:
                log_error(f"URLError: {e}")
                last_err = (base, e)
                continue
            except Exception as e:
                log_error(f"Exception: {e}")
                last_err = (base, e)
                continue

        if last_err:
            base, e = last_err
            if isinstance(e, HTTPError):
                raise e
            raise RuntimeError(f"Viimeinen kokeiltu osoite epäonnistui: {base} ({e})")
        raise RuntimeError("Haku epäonnistui tuntemattomasta syystä.")

    except HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")

        except Exception:
            body = ""
        msg = f"Haku epäonnistui: HTTP {e.code} {e.reason}"
        if body:
            msg += f"\nPalvelimen vastaus: {body[:500]}"
        log_error(msg)
        return msg

    except Exception as e:
        msg = f"Haku epäonnistui: {e}"
        log_error(msg)
        return msg
