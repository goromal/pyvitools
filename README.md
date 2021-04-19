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
- [scipy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [shapely](https://pypi.org/project/Shapely/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [open3d](http://www.open3d.org/docs/release/getting_started.html) (*optional*--for pointcloud visualization)
  - If using Python2.7, will have to compile from source. See link.
- ROS (*optional*--see below)

If you want the ROS-related functionality without installing ROS (e.g., if your OS isn't Linux), try utilizing this [standalone fork of select ROS python libraries](https://github.com/rospypi/simple):

```bash
pip install --extra-index-url https://rospypi.github.io/simple/ rospy std-msgs geometry-msgs sensor-msgs nav-msgs cv-bridge rosbag roslz4 actionlib-msgs
```

Another possible option found [here](https://discourse.ros.org/t/experimental-python-package-index-for-ros/10366/2).

