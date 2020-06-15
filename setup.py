from setuptools import setup

setup(name="video_tools",
      version='0.2.1',
      description='tools for anayzing video mostly using skvideo.',
      url="https://github.com/jpcurrea/video_tools.git",
      author='Pablo Currea',
      author_email='johnpaulcurrea@gmail.com',
      license='MIT',
      packages=['video_tools'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'sk-video',
          'fly_eye',
          'bird_call'
      ],
      zip_safe=False)
