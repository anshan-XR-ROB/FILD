# Fast and Incremental Loop Closure Detection with Deep Features and Proximity Graphs

This is the C++ implementation of our new article:
Shan An, Haogang Zhu, Dong Wei, and Konstantinos A. Tsintotas. Fast and Incremental Loop Closure Detection with Deep Features and Proximity Graphs. Submitted to Transactions on Robotics.

**Abstract:** 
In recent years, methods concerning the place recognition task have been extensively examined from the robotics community within the scope of simultaneous localization and mapping applications.
In this article, an appearance-based loop closure detection pipeline is proposed, entitled ``FILD++" (Fast and Incremental Loop closure Detection).
When the incoming camera observation arrives, global and local visual features are extracted through two passes of a single convolutional neural network.
Subsequently, a modified hierarchical-navigable small-world graph incrementally generates a visual database that represents the robot's traversed path based on the global features.
Given the query sensor measurement, similar locations from the trajectory are retrieved using these representations, while an image-to-image pairing is further evaluated thanks to the spatial information provided by the local features.
Exhaustive experiments on several publicly-available datasets exhibit the system's high performance and low execution time compared to other contemporary state-of-the-art pipelines.

