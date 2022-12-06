#include <stdio.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
 
// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{9,5}; 
 
int main() {
    // Creating vector to store vectors of 3D points for each checkerboard image
    vector<vector<cv::Point3f> > objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    vector<vector<cv::Point2f> > imgpoints;

    // Defining the world coordinates for 3D points
    vector<cv::Point3f> objp;
    for(int i{0}; i<CHECKERBOARD[1]; i++)
    {
        for(int j{0}; j<CHECKERBOARD[0]; j++)
        objp.push_back(cv::Point3f(j,i,0));
    }

    string img_path = "/works/find_chessboard/sample.jpg";
    cv::Mat frame, gray, undistort_img;

    // vector to store the pixel coordinates of detected checker board corners 
    vector<cv::Point2f> corner_pts;
    bool success;

    // Looping over all the images in the directory
    frame = cv::imread(img_path);
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        
    /* 
        * If desired number of corner are detected,
        * we refine the pixel coordinates and display 
        * them on the images of checker board
    */
    // cout << boolalpha << success << endl;
    if(success)
    {

        cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 5, 0.0000001);
        
        // refining pixel coordinates for given 2d points.
        cv::cornerSubPix(gray, corner_pts, cv::Size(11,11), cv::Size(-1,-1), criteria);
        
        // Displaying the detected corner points on the checker board
        cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
        cv::imwrite("chess.jpg", frame);
        
        objpoints.push_back(objp);
        imgpoints.push_back(corner_pts);
    }

    cv::Mat cameraMatrix,distCoeffs,R,T;

    /*
    * Performing camera calibration by 
    * passing the value of known 3D points (objpoints)
    * and corresponding pixel coordinates of the 
    * detected corners (imgpoints)
    */
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

    cout << "cameraMatrix : " << cameraMatrix << endl;
    cout << "distCoeffs : " << distCoeffs << endl;
    cout << "Rotation vector : " << R << endl;
    cout << "Translation vector : " << T << endl;

    undistort(frame, undistort_img, cameraMatrix, distCoeffs);
    cv::imwrite("undistort.jpg", undistort_img);

    return 0;
}