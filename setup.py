"""Setup configuration for the PoP package."""

from setuptools import find_packages, setup

setup(
    name="pop-framework",
    version="0.1.0",
    description=(
        "Proof-of-Perception: Certified Tool-Using Multimodal Reasoning "
        "with Compositional Conformal Guarantees"
    ),
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
