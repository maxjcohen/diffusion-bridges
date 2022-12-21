import setuptools

setuptools.setup(
    name="diffusion-bridges-maxjcohen",
    version="0.0.1",
    author="Max Cohen",
    author_email="research.riavr@simplelogin.co",
    description="A diffusion probabilistic model framework.",
    url="https://github.com/maxjcohen/diffusion-bridges",
    packages=["ddpm"],
    python_requires=">=3.6",
    install_requires=[
        "torch",
    ],
)
