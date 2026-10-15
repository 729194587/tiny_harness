"""Run the installed official grader, denying its automatic image-pull fallback.

Used only in the explicitly authorized grading subprocess. Grading logic and
arguments are the official package's; no installed package files are modified.
"""
import runpy


def main():
    from docker.models.images import ImageCollection

    def deny_pull(*args, **kwargs):
        raise RuntimeError("Only preloaded, pinned local evaluation images are allowed")

    original = ImageCollection.pull
    ImageCollection.pull = deny_pull
    try:
        runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
    finally:
        ImageCollection.pull = original


if __name__ == "__main__":
    main()
