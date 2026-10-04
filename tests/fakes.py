"""Test doubles shared by several test files (import them from here, not from conftest)."""


class FakeComfy:
    """Records uploads and graphs; every render returns one small PNG."""
    uploads: list = []
    graphs: list = []

    def __init__(self, url):
        pass

    def upload_image(self, path, subfolder="x"):
        from PIL import Image
        with Image.open(path) as im:
            FakeComfy.uploads.append((path.name, im.size, im.mode))
        return f"{subfolder}/{path.name}"

    def queue(self, graph):
        FakeComfy.graphs.append(graph)
        return "pid"

    def wait(self, ids, should_stop=None):
        return {"pid": {"outputs": {}}}

    def fetch_images(self, hist, node):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        return [buf.getvalue()]


class FakeLibrary:
    """Stands in for LoraLibrary: index() is all with_triggers and the pickers read."""

    def __init__(self, index: dict):
        self._index = index

    def index(self) -> dict:
        return self._index
