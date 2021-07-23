import setuptools


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setuptools.setup(
    name="various_gan_models",
    version="1.0.0",
    author="Nakai Yu.",
    description="various gan models",
    long_description=_requires_from_file('README.md'),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9.2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=_requires_from_file('requirements.txt'),
)