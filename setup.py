from setuptools import setup, find_packages

setup(
    name='pothole-detector',
    version='1.0',
    description='Streamlit app for pothole detection',
    author='Your Name',
    author_email='your_email@example.com',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'opencv-python-headless',
        'requests',
        'ultralytics',
    ],
)
