# Introduction

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue.svg)](https://pypi.org/project/peekingduck/)
[![PyPI version](https://badge.fury.io/py/peekingduck.svg)](https://pypi.org/project/peekingduck/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/peekingduck)](https://pypi.org/project/peekingduck/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/aimakerspace/PeekingDuck/blob/dev/LICENSE)

## What is PeekingDuck?

PeekingDuck is an open-source, modular framework in Python, built for Computer Vision (CV) inference. It helps to significantly cut down development time when building CV pipelines. The name "PeekingDuck" is a play on these words: "Peeking" in a nod to CV; and "Duck" in [duck typing](https://en.wikipedia.org/wiki/Duck_typing).


## How PeekingDuck Works

**Nodes** are the building blocks of PeekingDuck. Each node is a wrapper for a Python function, and contains information on how other PeekingDuck nodes may interact with it.

PeekingDuck has 5 types of nodes:

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/diagrams/node_types.drawio.svg">

A **pipeline** governs the behavior of a chain of nodes. The diagram below shows the pipeline used in the previous demo. Nodes in a pipeline are called in sequential order, and the output of one node will be the input to another. For example, `input.live` produces "img", which is taken in by `model.yolo`, and `model.yolo` produces "bboxes", which is taken in by `draw.bbox`. For ease of visualisation, not all the inputs and outputs of these nodes are included in this diagram.

<img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/diagrams/yolo_demo.drawio.svg">


## Explore PeekingDuck's Features

The rest of this webpage contains more details about using PeekingDuck, including information on:
- [Changing PeekingDuck nodes](getting_started/02_configure_pkdk) and their settings
- [Official documentation](/peekingduck.pipeline.nodes) for all PeekingDuck nodes, describing their behaviour, inputs, outputs and settings
- Creating your own [custom nodes](getting_started/03_custom_nodes), and using them with PeekingDuck nodes
- Using PeekingDuck as an imported Python module within your project, via the [Python API](getting_started/04_python_mode)
- `model` nodes [FPS and accuracy benchmarks](resources/01_benchmarks), as well as details about [keypoints and class IDs](resources/02_model_indices)

You are also welcome to join discussions about using PeekingDuck in the following channels:
- [Github discussion board](https://github.com/aimakerspace/PeekingDuck/discussions)
- [AI Singapore's Community forum](https://community.aisingapore.org/groups/computer-vision/forum/)


## PeekingDuck Use Cases

AI models are cool and fun, but we're even more interested to use them to solve real-world problems. We've combined dabble nodes with model nodes to create **use cases**, such as [social distancing](https://aisingapore.org/2020/06/hp-social-distancing/) and [group size checking](https://aisingapore.org/2021/05/covid-19-stay-vigilant-with-group-size-checker/) to help combat COVID-19. For more details, click on the heading of each use case below.

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Social Distancing](use_cases/social_distancing.md)          | [Zone Counting](use_cases/zone_counting.md)                  |
| <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/social_distancing.gif" width="100%"> | <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/zone_counting.gif" width="100%"> |
| [Group Size Checking](use_cases/group_size_checking.md)      | [Object Counting](use_cases/object_counting.md)              |
| <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/group_size_check_2.gif" width="100%"> | <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/object_counting.gif" width="100%"> |
| [Privacy Protection (Faces)](use_cases/privacy_protection_faces.md) | [Privacy Protection (License Plates)](use_cases/privacy_protection_license_plate.md) |
| <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/privacy_protection_faces.gif" width="100%"> | <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/privacy_protection_license_plates.gif" width="100%"> |
| [Face Mask Detection](use_cases/face_mask_detection.md)      |                                                              |
| <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/mask_detection.gif" width="100%"> |                                                              |
|                                                              |                                                              |

We're constantly developing new nodes to increase PeekingDuck's capabilities. You've gotten a taste of some of our commonly used nodes in the previous demos, but PeekingDuck can do a lot more. To see what other nodes are available, check out PeekingDuck's [API Reference](/peekingduck.pipeline.nodes).


## Acknowledgements

This project is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG-RP-2019-050). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.


## License

PeekingDuck is under the open source [Apache License 2.0](https://github.com/aimakerspace/PeekingDuck/blob/dev/LICENSE) (:

Even so, your organisation may require legal proof of its right to use PeekingDuck, due to circumstances such as the following:
- Your organisation is using PeekingDuck in a jurisdiction that does not recognise this license
- Your legal department requires a license to be purchased
- Your organisation wants to hold a tangible legal document as evidence of the legal right to use and distribute PeekingDuck

[Contact us](https://aisingapore.org/home/contact/) if any of these circumstances apply to you.
