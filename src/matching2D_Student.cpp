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
        // ...
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);  
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2); 
        double minDescDistRatio = 0.8;
        for (auto nn = knn_matches.begin(); nn != knn_matches.end(); ++ nn)
        {

            if ((*nn)[0].distance < minDescDistRatio * (*nn)[1].distance)
            {
                matches.push_back((*nn)[0]);
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
    else
    {

        //...
        descKeypointsOtherType(extractor, keypoints, img, descriptors, descriptorType);  
    }

    // perform feature description
    // double t = (double)cv::getTickCount();
    // extractor->compute(img, keypoints, descriptors);
    // t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    if ((descriptorType.compare("ORB")==0) || (descriptorType.compare("AKAZE")==0) )
        extractor->detectAndCompute(img, cv::Mat(),keypoints, descriptors);
    else
        extractor->compute(img, keypoints, descriptors);
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
    // t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

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

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, string detectorType, bool bVis)
{
    if (detectorType.compare("HARRIS") == 0)
    {
        detKeypointsHarris(keypoints,img,bVis);
    }
    else if (detectorType.compare("FAST") == 0)
    {
        detKeypointsFast(keypoints,img,bVis);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detKeypointsBrisk(keypoints,img,bVis);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detKeypointsOrb(keypoints,img,bVis);
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detKeypointsAkaze(keypoints,img,bVis);
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detKeypointsSift(keypoints,img,bVis);
    }
    
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // initialize parameters
    int blockSize = 2;     
    int apertureSize = 3;  
    int minResponse = 100; 
    double k = 0.04;

    // detect Harris corners and normalize outputs
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    
    // max permissible overlap between two features in %, used during non-maxima suppression
    double maxOverlap = 0.0; 
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { 
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                     
                            *it = newKeyPoint; 
                            break;
                        }
                    }
                }
                if (!bOverlap)
                {   // store new keypoint in dynamic list                                  
                    keypoints.push_back(newKeyPoint); 
                }
            }
        } 
    } 


    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(40);
    fast->detect(img, keypoints);

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "FAST Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(); 
    brisk->detect(img,keypoints);

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,bool bVis)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detect(img, keypoints);

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "ORB Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detect(img, keypoints);    

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "AKAZE Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    sift->detect(img,keypoints);

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "SIFT Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void descKeypointsOtherType(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
    
    if(descriptorType.compare("BRIEF")==0)
    {
        descKeypointsBrief(extractor,keypoints,img,descriptors);
    }
    else if (descriptorType.compare("ORB")==0)
    {
        descKeypointsOrb(extractor,keypoints,img,descriptors);
    }
    else if (descriptorType.compare("FREAK")==0)
    {
        descKeypointsFreak(extractor, keypoints,img,descriptors);
    }
    else if (descriptorType.compare("AKAZE")==0)
    {
        descKeypointsAkaze(extractor, keypoints,img,descriptors);
    }
    else if (descriptorType.compare("SIFT")==0)
    {
        descKeypointsSift(extractor,keypoints,img,descriptors);
    }
    
}

void descKeypointsBrisk(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors)
{
    int 	thresh = 30;
    int 	octaves = 3;
    float 	patternScale = 1.0f;
    extractor = cv::BRISK::create(thresh,octaves,patternScale);
}

void descKeypointsBrief(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors)
{
    // initialize parameters
    int bytes = 32;
    bool use_orientation = false;

    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes,use_orientation);
}
void descKeypointsOrb(cv::Ptr<cv::DescriptorExtractor>& extractor, std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors)
{
    // initialize parameters
    int 	nfeatures = 500;
    float 	scaleFactor = 1.2f;
    int 	nlevels = 8;
    int 	edgeThreshold = 31;
    int 	firstLevel = 0;
    int 	WTA_K = 2;
    auto 	scoreType = cv::ORB::HARRIS_SCORE;
    int 	patchSize = 31;
    int 	fastThreshold = 20;

    extractor = cv::ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
}
void descKeypointsFreak(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors)
{
    // initialize parameters
    bool 	orientationNormalized = true;
    bool 	scaleNormalized = true;
    float 	patternScale = 22.0f;
    int 	nOctaves = 4;
    const std::vector< int > & 	selectedPairs = std::vector< int >();

    extractor = cv::xfeatures2d::FREAK::create(orientationNormalized,scaleNormalized,patternScale,nOctaves,selectedPairs);
}
void descKeypointsAkaze(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors)
{
    // initialize parameters
    auto 	descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int 	descriptor_size = 0;
    int 	descriptor_channels = 3;
    float 	threshold = 0.001f;
    int 	nOctaves = 4;
    int 	nOctaveLayers = 4;
    int 	diffusivity = cv::KAZE::DIFF_PM_G2;
    extractor = cv::AKAZE::create(descriptor_type,descriptor_size,descriptor_channels,threshold,nOctaves,nOctaveLayers,diffusivity);
}
void descKeypointsSift(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img,cv::Mat &descriptors)
{
    // initialize parameters
    int 	nfeatures = 0;
    int 	nOctaveLayers = 3;
    double 	contrastThreshold = 0.04;
    double 	edgeThreshold = 10;
    double 	sigma = 1.6;
    
    extractor = cv::xfeatures2d::SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);

}
