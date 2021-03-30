# pyvitools

Python tools for processing and analyzing data for visual-inertial estimation applications.

Provides functions and classes for:

- Algebraic anipulation of vectors and SO(3)/SE(3) objects.
- Differentiating and integrating the above.
- ROSbag processing.
- Dataset manipulation and frame transformations.
- (Coming soon) Plotting of all the above.

**[API Documentation](https://goromal.github.io/pyvitools/)**

## Dependencies

- [numpy](https://numpy.org/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- ROS (*optional*--see below)

If you want the ROS-related functionality without installing ROS (e.g., if your OS isn't Linux), try utilizing this [standalone fork of select ROS python libraries](https://github.com/rospypi/simple):

```bash
pip install --extra-index-url https://rospypi.github.io/simple/ rospy std-msgs geometry-msgs sensor-msgs nav-msgs cv-bridge rosbag roslz4
```

Another possible option found [here](https://discourse.ros.org/t/experimental-python-package-index-for-ros/10366/2).

