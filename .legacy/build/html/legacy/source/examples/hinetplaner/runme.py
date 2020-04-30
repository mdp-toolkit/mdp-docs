
import webbrowser
import threading
import SimpleHTTPServer
import posixpath

import jsonrpc
import hinetplaner

FILE = 'frontend.xhtml'
HOST = ''  # or 'localhost'
PORT = 8080


class MySimpleJSONRPCRequestHandler(jsonrpc.SimpleJSONRPCRequestHandler,
                                    SimpleHTTPServer.SimpleHTTPRequestHandler):
    """Add GET handling to serve the initial page."""
    
    # fix missing xhtml mime type in mimetypes.py
    def guess_type(self, path):
        base, ext = posixpath.splitext(path)
        if ext == ".xhtml":
            return "application/xhtml+xml"
        else:
            return SimpleHTTPServer.SimpleHTTPRequestHandler.guess_type(self,
                                                                        path)


def open_browser():
    """Start a browser after waiting for half a second."""
    def _open_browser():
        webbrowser.open('http://localhost:%s/%s' % (PORT, FILE))
    thread = threading.Timer(0.5, _open_browser)
    thread.start()

def start_server():
    """Start the server."""
    server = jsonrpc.SimpleJSONRPCServer(
                                (HOST, PORT),
                                requestHandler=MySimpleJSONRPCRequestHandler)
    server.register_function(hinetplaner.get_layer_params)
    server.register_function(hinetplaner.get_hinet)
    server.register_function(hinetplaner.get_layer_coverage)
    server.register_function(hinetplaner.get_available_hinet_configs)
    server.register_function(hinetplaner.save_hinet_config)
    server.register_function(hinetplaner.get_hinet_config)
    server.serve_forever()
    
if __name__ == "__main__":
    open_browser()
    start_server()
