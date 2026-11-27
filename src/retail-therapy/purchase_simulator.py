"""simple localhost page showing how a GUI calls the prediction model"""
import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from string import Template
from urllib.parse import parse_qs

import numpy as np

from next_category import MODEL_PATH, load_model, predict_next

PAGE_PATH = Path(__file__).with_suffix(".html")
MAX_REQUEST_BYTES = 1_048_576


def add_purchase(history, category_id):
    """append a category in purchase order, repeats are allowed"""
    if (
        isinstance(category_id, bool)
        or not isinstance(category_id, (int, np.integer))
        or not 0 <= category_id < 148
    ):
        raise ValueError("Choose a category from the dropdown")
    history.append(int(category_id))


def prepare_inputs(history):
    """convert ordered categories into the arrays predict_next expects"""
    if not isinstance(history, list) or any(
        type(category_id) is not int or not 0 <= category_id < 148
        for category_id in history
    ):
        raise ValueError("History must be a list of category IDs from 0 through 147")
    category_ids = np.full((len(history), 3), 255, dtype=np.uint8)
    category_weights = np.zeros((len(history), 3), dtype=np.float32)

    # selected category is known, unused positions stay missing
    category_ids[:, 0] = history
    category_weights[:, 0] = 1.0
    return category_ids, category_weights


def predict_history(model, history):
    """this is the handoff from the GUI to the existing model"""
    category_ids, category_weights = prepare_inputs(history)
    return predict_next(model, category_ids, category_weights)


def top_predictions(model, probabilities):
    """get five category names and their original probabilities"""
    names = {
        category_id: name
        for name, category_id in model["metadata"]["taxonomy"].items()
    }
    category_ids = np.argsort(-probabilities, kind="stable")[:5]
    return [
        (names[int(category_id)], float(probabilities[category_id]))
        for category_id in category_ids
    ]


def prediction_result(model, history):
    """return predictions only after a purchase has been added"""
    if not history:
        return {"history": [], "top5": []}
    probabilities = predict_history(model, history)
    taxonomy = model["metadata"]["taxonomy"]
    names = {category_id: name for name, category_id in taxonomy.items()}
    return {
        "history": [names[category_id] for category_id in history],
        "top5": [
            {"category": name, "probability": probability}
            for name, probability in top_predictions(model, probabilities)
        ],
    }


def render_page(model, history, selected=0, error=""):
    """fill the plain HTML form and names-only output"""
    taxonomy = model["metadata"]["taxonomy"]
    names = {category_id: name for name, category_id in taxonomy.items()}
    options = []
    for name, category_id in taxonomy.items():
        selection = ""
        if category_id == selected:
            selection = " selected"
        options.append(
            f'<option value="{category_id}"{selection}>{escape(name)}</option>'
        )

    # keep the list on error, clear predictions from the failed request
    result = {"history": [names[category_id] for category_id in history], "top5": []}
    if not error:
        result = prediction_result(model, history)
    template = Template(PAGE_PATH.read_text())
    return template.substitute(
        options="\n".join(options),
        history=escape(json.dumps(history), quote=True),
        history_output=escape(json.dumps(result["history"], indent=2, ensure_ascii=False)),
        predictions=escape(
            json.dumps(result["top5"], indent=2, ensure_ascii=False, allow_nan=False)
        ),
        error=escape(error),
    ).encode("utf-8")


class SimulatorHandler(BaseHTTPRequestHandler):
    def send_body(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_page(self, status=200, history=None, selected=0, error=""):
        if history is None:
            history = []
        try:
            body = render_page(self.server.model, history, selected, error)
        except Exception as failure:
            status = 500
            error = f"Prediction failed: {failure}"
            body = render_page(self.server.model, history, selected, error)
        self.send_body(status, body, "text/html; charset=utf-8")

    def do_GET(self):
        if self.path == "/":
            self.send_page()
        else:
            self.send_body(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/":
            self.send_body(404, b"Not found", "text/plain; charset=utf-8")
            return
        if self.headers.get_content_type() != "application/x-www-form-urlencoded":
            self.send_page(415, error="Expected an HTML form")
            return
        history = []
        selected = 0
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("Form must be between 1 byte and 1 MiB")
            form = parse_qs(
                self.rfile.read(length).decode("utf-8"),
                keep_blank_values=True,
                strict_parsing=True,
            )
            if any(len(values) != 1 for values in form.values()):
                raise ValueError("Expected one value per form field")

            # hidden history is input too, validate before using it
            submitted = json.loads(form["history"][0])
            prepare_inputs(submitted)
            history = submitted
            category_id = int(form["category"][0])
            if not 0 <= category_id < 148:
                raise ValueError("Choose a category from the dropdown")
            selected = category_id
            action = form["action"][0]
            if action == "add":
                add_purchase(history, selected)
            elif action == "undo":
                if history:
                    history.pop()
            elif action == "clear":
                history.clear()
            else:
                raise ValueError("Unknown form action")
        except (ValueError, KeyError, UnicodeError) as error:
            self.send_page(400, history, selected, str(error))
            return
        self.send_page(200, history, selected)


def create_server(model, port=8000):
    # one shared model, history travels with each form
    server = HTTPServer(("127.0.0.1", port), SimulatorHandler)
    server.model = model
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")

    # load once at startup, never train from the webpage
    try:
        model = load_model(args.model)
    except Exception as error:
        parser.exit(
            1,
            f"Cannot load model {args.model}: {error}\n"
            "Train with: python src/retail-therapy/next_category.py --full\n",
        )
    try:
        server = create_server(model, args.port)
    except OSError as error:
        parser.exit(1, f"Cannot start localhost server: {error}\nTry another --port.\n")
    with server:
        print(f"Open http://127.0.0.1:{args.port} (Ctrl-C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
