# SFND_3D_Object_Tracking
Camera and Lidar fusion for object tracking 

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* Point Cloud Library >= 1.2  

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.


# My Project Report

## FP.1 Task : Match 3D Objects
I created for loop over all matches then I seperated matches into the query and train points. When if a bounding box of previous frame contains the query point, I checked all bounding boxes of current frames for querying which one contains respective train point. After, I counted points which is contained and I stored them in 2D vector.
```c++
vector<vector<int>> matchesStoreMatrix(prevFrame.boundingBoxes.size(),vector<int>(currFrame.boundingBoxes.size(),0));
for (int i = 0; i < matches.size(); i++)
{
   auto tempPrevKeyPoint = prevFrame.keypoints[matches[i].queryIdx].pt;
   auto tempCurrentKeyPoint = currFrame.keypoints[matches[i].trainIdx].pt;
        
   for (int j = 0; j < prevFrame.boundingBoxes.size(); j++)
   {
     if(prevFrame.boundingBoxes[j].roi.contains(tempPrevKeyPoint))   
     {
       for (int k = 0; k < currFrame.boundingBoxes.size(); k++)
       {
         if(currFrame.boundingBoxes[k].roi.contains(tempCurrentKeyPoint))
         {
           matchesStoreMatrix[j][k] = matchesStoreMatrix[j][k] + 1;
         }
```

Then I created another nested for loop over all previous frames and current frames and I checked what coordinates of 2D vector has max count. Then I assigned these indexes to bbBestMatches.

```c++
for (int i = 0; i < prevFrame.boundingBoxes.size(); i++)
    {
        int maxMatches = 0;
        int maxMatchesID = -1;
        for (int k = 0; k < currFrame.boundingBoxes.size(); k++)
        {
             if(matchesStoreMatrix[i][k] > maxMatches)
             {
                 maxMatches = matchesStoreMatrix[i][k];
                 maxMatchesID = k;
             }
        }
        bbBestMatches.emplace(i,maxMatchesID);
    }
```

## FP.2 Task : Compute Lidar Based TTC
I filled the computeTTCLidar function to compute time to collison according to nearest x point for only car which the car in front of our car. Also lidar data has some noise/outlier. For filtering this lidar data, I created new function which name is removeLidarOutliersStatisticaly with PointCloud library based on mean KNN distance. This funciton take lidar data, meanK value and standart deviation (std) as input. Then it returns filtered lidar data. meanK value means that number of neighbors to analyze for each point. I setted value of meanK to 50 and value of std to 1. If a point who have a distance larger than 1 standard deviation of the mean distance to the query point is going to marked outlier and remove.

```c++
pcl::PointCloud<pcl::PointXYZ>::Ptr  removeLidarOutliersStatisticaly(const std::vector<LidarPoint> &lidarPoints,int meanK,float stdDev)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr lidarCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredLidarCloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (int i=0; i<lidarPoints.size(); i++)
    {
      lidarCloud>push_back(pcl::PointXYZ((float)lidarPoints[i].x,(float)lidarPoints[i].y,(float)lidarPoints[i].z));
    }

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (lidarCloud);
    sor.setMeanK (meanK);
    sor.setStddevMulThresh (stdDev);
    sor.filter (*filteredLidarCloud);
    return filteredLidarCloud;
}
```
Later I used this function in computeTTCLidar function as;

```c++
auto filteredPrevLidar = removeLidarOutliersStatisticaly(lidarPointsPrev,50,1.0);
auto filteredCurrLidar = removeLidarOutliersStatisticaly(lidarPointsCurr,50,1.0);
```

## FP.3 Task : Associate Keypoint Correspondences with Bounding Boxes
Firstly, I assigned all matches to respective bounding box according to containing the keypoints of matches. Then, I filtered these matches using mean Euclidean distance.

```c++
euclideanDist = sqrt(pow((kptsCurr.at(boundingBox.kptMatches[i].trainIdx).pt.x - kptsPrev.at(boundingBox.kptMatches[i].queryIdx).pt.x),2.0) + pow((kptsCurr.at(boundingBox.kptMatches[i].trainIdx).pt.y - kptsPrev.at(boundingBox.kptMatches[i].queryIdx).pt.y),2.0));
```

## FP.4 Task : Compute Camera-based TTC
I created a function for finding detected keypoints distance ratio respective bounding boxes to compute time to collision with successive images. To eliminate false matches I used median filter.

```c++
std::sort(distRatios.begin(), distRatios.end());
double medianDistRatio = 0.0;
if(distRatios.size()%2 != 0) //distRatio size is even
{
   medianDistRatio = distRatios[(int)std::ceil(distRatios.size() / 2)];
}
else if(distRatios.size()%2 == 0)
{
   medianDistRatio = (distRatios[distRatios.size()/2] + distRatios[(distRatios.size()/2) + 1]) / 2.0;
}
```

## FP.5 Task : Lidar Performance Evaluation 
I faced 2 abnormal values (negative values) in process of TTC with lidar for two frames without noise filtering. In my opinion that’s happened because lidar has 360 rotation degree. So some frames include outliers which came from the car which behind of us. When these lidar points projected into the 2D image plane, the cluster of outliers corresponds to the bounding box coordinates of preceding vehicle. If I use my outlier removing filter, negative values are gone. I want to show one example of two abnormal situation.



