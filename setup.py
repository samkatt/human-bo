from setuptools import setup

setup(
    name="human-in-the-bo",
    packages=["human_bo"],
    scripts=[
        "scripts/run_human_ai_experiment.py",
        "scripts/visualize_human_ai_experiment.py",
    ],
)
