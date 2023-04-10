import setuptools
import os

def read_requirements():
    build_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(build_dir, "yolov5", "requirements.txt")) as f:
        return f.read().splitlines()


# default variable to be overwritten by the version.py file
version = "unknown"
# load and overwrite version and release info from version.py
try:
    exec(open(os.path.join("utils", "neuralmagic", "version.py")).read())
except:
    # in build.sh utils may be nested under yolov5/
    exec(open(os.path.join("yolov5", "utils", "neuralmagic", "version.py")).read())

setuptools.setup(
    name="nm-yolov5" if is_release else "nm-yolov5-nightly",
    version=version,  # major.minor.patch to match NM repos, fourth entry is either yolov5 base version or nightly date
    author="",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neuralmagic/yolov5",
    packages=['yolov5', 'yolov5.models', 'yolov5.utils'],
    python_requires=">=3.6",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={'': ['yolov5/models/*.yaml', 'yolov5/data/*']},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
