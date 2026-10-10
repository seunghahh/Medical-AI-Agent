"""Read-only localhost monitor; the CLI publishes events without waiting for a browser."""
import json
import os
import sys
import subprocess
import webbrowser
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from .data import ROOT

LIVE = Path(os.getenv('CLINIC_LIVE_DIR', ROOT / 'outputs' / '.live'))
WEB = ROOT / 'web'


class Journal:
    def __init__(self, output, metadata, directory=LIVE):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = output / 'events.jsonl'
        self.stream = self.path.open('w', encoding='utf-8')
        self.run_id = uuid4().hex
        self.sequence = 0
        self.emit('run_start', run_label=output.name, **metadata)
        pointer = self.directory / f'{os.getpid()}.tmp'
        pointer.write_text(json.dumps({'run_id': self.run_id, 'path': str(self.path.resolve()), 'pid': os.getpid()}))
        pointer.replace(self.directory / 'latest.json')

    def emit(self, kind, **data):
        if self.stream.closed:
            return
        self.sequence += 1
        event = dict(data, type=kind, seq=self.sequence, time=time.time(), run_id=self.run_id)
        try:
            self.stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            self.stream.flush()
        except OSError as exc:
            print(f'Live monitor disabled: {exc}', file=sys.stderr)
            self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.emit('run_end', status='interrupted' if kind is KeyboardInterrupt else 'failed' if error else 'finished', error=str(error) if error else None)
        self.stream.close()


def snapshot(run_id='', after=0, directory=LIVE):
    try:
        pointer = json.loads((Path(directory) / 'latest.json').read_text())
    except FileNotFoundError:
        return {'run_id': None, 'events': [], 'alive': False}
    # The local CLI owns this pointer; HTTP callers cannot supply a filesystem path.
    events = []
    if not Path(pointer['path']).is_file():
        return {'run_id': None, 'events': [], 'alive': False}
    with Path(pointer['path']).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # A concurrent writer has not flushed a complete event yet.
            event = json.loads(line)
            if run_id != pointer['run_id'] or event['seq'] > after:
                events.append(event)
    try:
        os.kill(pointer['pid'], 0)
        alive = True
    except ProcessLookupError:
        alive = False
    return {'run_id': pointer['run_id'], 'events': events, 'alive': alive}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        try:
            if url.path == '/api/health':
                body = json.dumps({'app': 'nova-live-clinic', 'root': str(ROOT), 'live': str(LIVE)}).encode()
                mime = 'application/json; charset=utf-8'
            elif url.path == '/api/events':
                query = parse_qs(url.query)
                after = max(0, int(query.get('after', ['0'])[0]))
                body = json.dumps(snapshot(query.get('run', [''])[0], after), ensure_ascii=False).encode()
                mime = 'application/json; charset=utf-8'
            else:
                assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                          '/motion.mjs': ('motion.mjs', 'text/javascript; charset=utf-8'),
                          '/style.css': ('style.css', 'text/css; charset=utf-8'),
                          '/room.jpg': ('room.jpg', 'image/jpeg'),
                          '/sprites.png': ('sprites.png', 'image/png')}
                if url.path not in assets:
                    self.send_error(404)
                    return
                filename, mime = assets[url.path]
                body = (WEB / filename).read_bytes()
        except ValueError:
            self.send_error(400)
            return
        except OSError:
            self.send_error(503, 'Live event file unavailable')
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def serve(port):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Live hospital: http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def monitor_ready(url):
    try:
        with urlopen(url + 'api/health', timeout=0.5) as response:
            identity = json.load(response)
    except HTTPError as exc:
        raise OSError('Live port is occupied by another or outdated server; stop it or choose --port.') from exc
    except URLError:
        return False
    except (ValueError, TimeoutError) as exc:
        raise OSError('Live port did not return a valid server identity.') from exc
    if identity != {'app': 'nova-live-clinic', 'root': str(ROOT), 'live': str(LIVE)}:
        raise OSError('Live port belongs to another project; choose a different --port.')
    return True


def open_monitor(port):
    url = f'http://127.0.0.1:{port}/'
    if not monitor_ready(url):
        LIVE.mkdir(parents=True, exist_ok=True)
        with (LIVE / 'server.log').open('a') as log:
            process = subprocess.Popen([sys.executable, str(ROOT / 'run.py'), 'serve', '--port', str(port)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        for _ in range(50):
            if monitor_ready(url):
                break
            if process.poll() is not None:
                raise OSError(f'Live server exited; see {LIVE / "server.log"}')
            time.sleep(0.1)
        else:
            raise OSError(f'Live server did not start; see {LIVE / "server.log"}')
    # Browsers may honor a new-window request as a new tab instead.
    try:
        opened = webbrowser.open(url, new=1, autoraise=True)
    except webbrowser.Error:
        opened = False
    if not opened:
        print(f'Open Live Clinic manually: {url}', file=sys.stderr)
    return url
