import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

__version__ = "0.0.0"

REPO_NAME = 'house-rent-prediction-model'
AUTHOR_USER_NAME = 'karthik4-4'
SRC_REPO = 'House_Rent_Prediction'

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="python package for project",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)