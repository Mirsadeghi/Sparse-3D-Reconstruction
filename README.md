
# Sparse-3D-Reconstruction

A sample project to do sparse 3D reconstruction of pair images. 
This project contains all steps of 3D reconstruction.

## Getting Started

In this project, all of these steps are includedL<br/>
1. Computing Fundamental matrix using SIFT and RANSAC<br/>
2. Image Rectification<br/>
3. Dense matching using block matcing<br/>
4. sparse 3D reconstruction using projection matrix<br/>
5. Outlier removal<br/>
6. creating mesh for point cloud<br/>

### Prerequisites

You need to hava some primeval information about concept of 3D image processing.


## Running the tests

simply run ```ACV_prj1().m```

## Authors

* **Ehsan Mirsadeghi** - *Initial work*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This is a final project of advanced 3D computer vision course prosented by [Prof. Shohreh Kasaei](http://sharif.edu/~kasaei/)


# Visual Results:

|  Rectification  | |
| --------------- | -------------- |
|  ![](Rec2.jpg)  |  ![](I2.jpg)   |
|  ![](Rec.jpg)   |![](RoIRec.jpg) |

|      Dense Macthing Results      | |
| --------------- | -------------- |
| ![](Dense1.jpg) |![](Dense2.jpg) |
|  ![](Dense3.jpg)| |

|Sparse reconstruction |
| -------------------- |
|![](sparse_result.jpg)|
