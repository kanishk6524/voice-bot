"""
Hugging Face Spaces entry point (app.py for Gradio-wrapped or plain Flask).

For Hugging Face Spaces using Docker or the SDK=gradio approach,
we expose the Flask app directly. HF Spaces runs on port 7860.

If deploying as a Docker space, this file is not needed — use Dockerfile instead.
For static Spaces the Flask app runs directly.
"""
# Re-export for HF compatibility
from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
