#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        //We should convert binary descriptors to float because of opencv bug
        if(descRef.type() != CV_32F)
        {
            descRef.convertTo(descRef,CV_32F);
        }
        if(descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource,CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { 
      // k nearest neighbors (k=2)
      //implement k-nearest-neighbor matching
      const float ratio_threshold = 0.8f;
      vector<vector<cv::DMatch>> knnMatches;
      matcher->knnMatch(descSource, descRef, knnMatches, 2);
      //filter matches using descriptor distance ratio test
      for (int i = 0; i < knnMatches.size(); i++)
      {
            if (knnMatches[i][0].distance < (ratio_threshold * knnMatches[i][1].distance))
            {
                 matches.push_back(knnMatches[i][0]);
            }

       }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        //Detection parameters if I need
        /*int 	nfeatures = 500;
	    float 	scaleFactor = 1.2f;
	    int 	nlevels = 8;
	    int 	edgeThreshold = 31;
	    int 	firstLevel = 0;
	    int 	WTA_K = 2;
	    int 	scoreType = 0; //cv::ORB::HARRIS_SCORE
	    int 	patchSize = 31;
	    int 	fastThreshold = 20;*/
        extractor = cv::ORB::create();
         
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        //Detection parameters if I need
        /*int 	descriptor_type = 5; // cv::AKAZE::DESCRIPTOR_MLDB
	    int 	descriptor_size = 0;
	    int 	descriptor_channels = 3;
	    float 	threshold = 0.001f;
	    int 	nOctaves = 4;
	    int 	nOctaveLayers = 4;
	    int 	diffusivity = 1; //cv::KAZE::DIFF_PM_G2*/
        extractor = cv::AKAZE::create();
         
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
         
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
         
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
      //Detection parameters if I need
      /*bool  orientationNormalized = true;
      bool  scaleNormalized = true;
      float patternScale = 22.0f;
      int   nOctaves = 4;*/
      extractor = cv::xfeatures2d::FREAK::create();
         
    }


    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    if(img.channels() == 3)
    {
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    }
   
    cv::Mat dst,dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(),CV_32FC1);
    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst,dst_norm,0,255,cv::NORM_MINMAX,CV_32FC1,cv::Mat());
    cv::convertScaleAbs(dst_norm,dst_norm_scaled);
    
    //overlap processing
	float overlap = 0.0;
	for (size_t i = 0; i < dst_norm.rows; i++)
	{
		for (size_t j = 0; j < dst_norm.cols; j++)
		{
			int matrix_response = (int)dst_norm.at<float>(i, j);
			if (matrix_response > minResponse)
			{
				cv::KeyPoint curr_KP;
				curr_KP.pt = cv::Point(j, i);
				curr_KP.size = 2 * apertureSize;
				curr_KP.response = matrix_response;

				bool b_overlap = false;
				for (auto it = keypoints.begin(); it != keypoints.end(); ++it) //compare other KPs to current KP
				{
					float kpt_overlap = cv::KeyPoint::overlap(curr_KP, *it);
					if (kpt_overlap > overlap)
					{
						b_overlap = true;
						if (curr_KP.response > (*it).response) {
							*it = curr_KP; //replace old KP with the current one
							break;
						}
					}
				}
				if (!b_overlap)
				{
					keypoints.push_back(curr_KP); // only add the KP if no overlap was found
				}
			}
		}
	}
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "HARRIS detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    //Visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "HARRIS Feature Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}

void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int threshold = 30;
    bool isNonMaximaSupression = true;
  
    if(img.channels() == 3)
    {
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    }
	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold,isNonMaximaSupression);
	double t = (double)cv::getTickCount();
	detector->detect(img, keypoints, cv::Mat());
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "FAST Detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  if(bVis)
  {
    cv::Mat visFastImage = img.clone();
	cv::drawKeypoints(img, keypoints, visFastImage,       cv::Scalar::all(-1));
	string WindowName = "FAST Feature Detector";
	cv::namedWindow(WindowName, 1);
	imshow(WindowName, visFastImage);
	cv::waitKey(0);
  }
	
}

void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    //BRISK parameters
    int thresh = 30;
	int octaves = 3;
	float patternScale = 1.0f;
  
    if(img.channels() == 3)
    {
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    }
  
    double t = (double)cv::getTickCount();
	cv::Ptr<cv::BRISK> detector = cv::BRISK::create(thresh,octaves,patternScale);
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "BRISK Detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
  //Visualize Results
  if(bVis)
  {
    cv::Mat visBRISKImage = img.clone();
	cv::drawKeypoints(img, keypoints, visBRISKImage, cv::Scalar::all(-1));
	string WindowName = "BRISK Feature Detector";
	cv::namedWindow(WindowName, 1);
	imshow(WindowName, visBRISKImage);
	cv::waitKey(0);
  }
}

void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    if(img.channels() == 3)
    {
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    }
    
    double t = (double)cv::getTickCount();
    //when I try to change parameters I got build errors, that's why I used default values
	cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "ORB Detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  //Visualize Results
  if(bVis)
  {
    cv::Mat visORBImage = img.clone();
	cv::drawKeypoints(img, keypoints, visORBImage,cv::Scalar::all(-1));
	string WindowName = "ORB Feature Detector";
	cv::namedWindow(WindowName, 1);
	imshow(WindowName, visORBImage);
	cv::waitKey(0);
  }

}

void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    
    if(img.channels() == 3)
    {
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    }
    
    double t = (double)cv::getTickCount();
    //when I try to change parameters I got build errors, that's why I used default values
	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    detector->detect(img,keypoints);
  
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "AKAZE Detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  //Visualize Results
  if(bVis)
  {
    cv::Mat visAKAZEImage = img.clone();
	cv::drawKeypoints(img, keypoints, visAKAZEImage,cv::Scalar::all(-1));
	string WindowName = "AKAZE Feature Detector";
	cv::namedWindow(WindowName, 1);
	imshow(WindowName, visAKAZEImage);
	cv::waitKey(0);
  }
}

void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{ 
   if(img.channels() == 3)
    {
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    }
   double t = (double)cv::getTickCount();
   cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  //Visualize Results
  if(bVis)
  {
    cv::Mat visSIFTImage = img.clone();
	cv::drawKeypoints(img, keypoints, visSIFTImage,cv::Scalar::all(-1));
	string WindowName = "SIFT Feature Detector";
	cv::namedWindow(WindowName, 1);
	imshow(WindowName, visSIFTImage);
	cv::waitKey(0);
  }
}


void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    if(detectorType.compare("SIFT") == 0)
    {
        detKeypointsSIFT(keypoints,img,bVis);
    }

    if(detectorType.compare("AKAZE") == 0)
    {
        detKeypointsAKAZE(keypoints,img,bVis);
    }

    if(detectorType.compare("ORB") == 0)
    {
        detKeypointsORB(keypoints,img,bVis);
    }

    if(detectorType.compare("BRISK") == 0)
    {
        detKeypointsBRISK(keypoints,img,bVis);
    }

    if(detectorType.compare("FAST") == 0)
    {
        detKeypointsFAST(keypoints,img,bVis);
    }
}