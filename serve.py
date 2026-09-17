import argparse
import os
import re
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


class VideoRequestHandler(SimpleHTTPRequestHandler):
    range_pattern = re.compile(r"bytes=(\d*)-(\d*)$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def parse_request(self):
        for method in (b"GET ", b"HEAD "):
            method_index = self.raw_requestline.find(method)
            prefix = self.raw_requestline[:method_index]
            if method_index > 0 and all(byte < 32 for byte in prefix):
                self.raw_requestline = self.raw_requestline[method_index:]
                break
        return super().parse_request()

    def end_headers(self):
        path = self.translate_path(self.path.split("?", 1)[0])
        if os.path.isfile(path):
            if path.endswith((".html", ".htm")):
                self.send_header("Cache-Control", "no-cache")
            elif path.endswith((".mp4", ".webp", ".png", ".ttf", ".pdf")):
                self.send_header("Cache-Control", "public, max-age=604800")
            elif path.endswith((".css", ".js")):
                self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

    def send_head(self):
        self.byte_range = None
        path = self.translate_path(self.path)
        range_header = self.headers.get("Range")
        if not range_header or not os.path.isfile(path):
            return super().send_head()

        match = self.range_pattern.fullmatch(range_header.strip())
        if not match:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        file = open(path, "rb")
        size = os.fstat(file.fileno()).st_size
        start_text, end_text = match.groups()
        if start_text:
            start = int(start_text)
            end = min(int(end_text), size - 1) if end_text else size - 1
        elif end_text:
            length = min(int(end_text), size)
            start, end = size - length, size - 1
        else:
            file.close()
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        if start >= size or start > end:
            file.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.fstat(file.fileno()).st_mtime))
        self.end_headers()
        self.byte_range = (start, end)
        return file

    def copyfile(self, source, outputfile):
        try:
            byte_range = getattr(self, "byte_range", None)
            if not byte_range:
                shutil.copyfileobj(source, outputfile)
                return

            start, end = byte_range
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass


def main():
    parser = argparse.ArgumentParser(description="Serve the project with MP4 range request support.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.bind, args.port), VideoRequestHandler)
    print(f"Serving {PROJECT_ROOT} on http://{args.bind}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()