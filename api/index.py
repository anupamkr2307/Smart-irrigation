import os
import sys

# Set up paths so that backend modules can be imported seamlessly
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
backend_dir = os.path.join(root_dir, "backend")

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Tell config we are running in Vercel environment
os.environ.setdefault("VERCEL", "1")

from app import create_app
from app.config import ProductionConfig

# Expose WSGI application for Vercel Python runtime
app = create_app(ProductionConfig)

# Default handler for local execution
if __name__ == "__main__":
    app.run()
