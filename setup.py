import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rukip",
    version="0.0.5",
    author="Hao-Yuan Chen",
    author_email="truecirclely@gmail.com",
    description="An Embedded CKIP Rasa NLU Components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/circlelychen/rukip",
    python_requires=">=3.6",
    packages=['rukip',
              'rukip.tokenizer',
              'rukip.featurizer'],
    install_requires=[
        "rasa>=1.4.0",
        "ckiptagger[tf]>=0.0.19"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
