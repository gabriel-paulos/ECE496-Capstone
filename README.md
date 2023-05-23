Design Project for the requirements of the University Of Toronto's ECE496 course.

The project contains a CLI tool and associated web application that uses Machine-Learning models to automatically remove the filler word "umm" from audio and video inputs.

The master branch contains the CLI tool with some experimental additions; the b_UI branch contains the both the CLI tool and web application that can be locally deployed.

Support for Apple Silicon installation for Tensorflow model packages:
  - Using Conda: https://developer.apple.com/metal/tensorflow-plugin/
  - Alternatively it can be done using Homebrew (dependency versions may differ): https://medium.com/@sorenlind/tensorflow-with-gpu-support-on-apple-silicon-mac-with-homebrew-and-without-conda-miniforge-915b2f15425b
