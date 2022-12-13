#include <iostream>
#include <sstream>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <filesystem>
#include <variant>
#include <map>
#include <regex>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// vpi
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Image.h>
#include <vpi/LensDistortionModels.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Remap.h>
#include <vpi/algo/PerspectiveWarp.h>

// rapid json
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/ostreamwrapper.h"

using namespace std;
using namespace rapidjson;

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

struct Vpi_param {
    float vpi_k1      = 0.0;
    float vpi_k2      = 0.0;
    float x_scale     = 1.0;
    float y_scale     = 1.0;
    float x_rotate    = 0.0;
    float y_rotate    = 0.0;
    float zx_perspect = 0.0;
    float zy_perspect = 0.0;
    float x_pad       = 0.0;
    float y_pad       = 0.0;
    int   x_focus     = 0;
    int   y_focus     = 0;  
    int   x_resize    = 640;
    int   y_resize    = 360;
};



void make_VPICamera_Param(const cv::Size& imgSize, const Vpi_param& vpi_param, VPICameraIntrinsic& K, VPICameraExtrinsic& X, vector<double>& coeffs) {

    using Mat3 = cv::Matx<double, 3, 3>;
    Mat3 camMatrix = Mat3::eye();
    camMatrix(0, 0) = 10.0;
    camMatrix(1, 1) = 10.0;

    if (vpi_param.x_focus == 0)
        camMatrix(0, 2) = imgSize.width / 2.0;
    else
        camMatrix(0, 2) = vpi_param.x_focus;

    if (vpi_param.y_focus == 0)
        camMatrix(1, 2) = imgSize.height / 2.0;
    else
        camMatrix(1, 2) = vpi_param.y_focus;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            K[i][j] = camMatrix(i, j);
        }
    }

    X[0][0] = X[1][1] = X[2][2] = 1;
    X[0][0] = vpi_param.x_scale;
    X[1][1] = vpi_param.y_scale;

    // ? rotate
    X[0][1] = vpi_param.x_rotate;
    X[1][0] = vpi_param.y_rotate;

    // ? pad
    X[0][3] = vpi_param.x_pad;
    X[1][3] = vpi_param.y_pad;

    // ? z축
    X[2][0] = vpi_param.zx_perspect;
    X[2][1] = vpi_param.zy_perspect;

    coeffs[0] = vpi_param.vpi_k1;
    coeffs[1] = vpi_param.vpi_k2;
    coeffs[2] = 0;
    coeffs[3] = 0;
}

float distanceCalculate(float x1, float y1, float x2, float y2)
{
    float x = x1 - x2; // calculating number to square in next step
    float y = y1 - y2;
    float dist;

    dist = pow(x, 2) + pow(y, 2); // calculating Euclidean distance
    dist = sqrt(dist);

    return dist;
}


float make_dist_mse_loss(const vector<cv::Point2f>& corner_pts, int chessboard[2]) {

    // chessboard[0] : 9 (열)
    // chessboard[1] : 5 (행)
    // cout << chessboard[0] << " | " << chessboard[1] << endl;
    // cout << "-----------------------------" << endl;

    // for (int i = 0; i < static_cast<int>(corner_pts.size()); i++) {
    //     float x = corner_pts[i].x;
    //     float y = corner_pts[i].y;
    //     cout << "x : " << x << " | y : " << y << endl;
    // }

    float sum_of_distance = 0.0f;
    float sum_of_square_distance = 0.0f;
    float num_of_distance = 0.0f;

    // left_right distance
    for (int i = 0; i < chessboard[1]; i++) {
        for (int j = 0; j < chessboard[0]; j++) {
            int bdot_idx = i * chessboard[0] + j;
            int adot_idx = i * chessboard[0] + j + 1;
            if (adot_idx == chessboard[0] * (i + 1)) {
                continue;
            }
            float dist = distanceCalculate(corner_pts[bdot_idx].x, corner_pts[bdot_idx].y, corner_pts[adot_idx].x, corner_pts[adot_idx].y);
            sum_of_distance += dist;
            sum_of_square_distance += pow(dist, 2);
            num_of_distance += 1;
            // cout << bdot_idx << " | " << adot_idx << endl;
        }
        // cout << "-----------------------------" << endl;
    }

    // up_down distance
    for (int i = 0; i < chessboard[0]; i++) { // 9
        for (int j = 0; j < chessboard[1]; j++) { // 5
            int bdot_idx = j * chessboard[0] + i;
            int adot_idx = (j + 1) * chessboard[0] + i;
            if (adot_idx == chessboard[0] * chessboard[1] + i) {
                continue;
            }
            float dist = distanceCalculate(corner_pts[bdot_idx].x, corner_pts[bdot_idx].y, corner_pts[adot_idx].x, corner_pts[adot_idx].y);
            sum_of_distance += dist;
            sum_of_square_distance += pow(dist, 2);
            num_of_distance += 1;
            // cout << bdot_idx << " | " << adot_idx << endl;
        }
        // cout << "-----------------------------" << endl;
    }

    float avg_distance = sum_of_distance / num_of_distance;
    float square_mse = sum_of_square_distance/num_of_distance - pow(avg_distance, 2);

    float mse = 0.0f;
    if (square_mse > 0) mse = sqrt(square_mse);

    // pair sampling
    // float d0 = distanceCalculate(corner_pts[10].x, corner_pts[10].y, corner_pts[11].x, corner_pts[11].y);
    // float d1 = distanceCalculate(corner_pts[11].x, corner_pts[11].y, corner_pts[20].x, corner_pts[20].y);
    // float d2 = distanceCalculate(corner_pts[20].x, corner_pts[20].y, corner_pts[21].x, corner_pts[21].y);
    // float d3 = distanceCalculate(corner_pts[21].x, corner_pts[21].y, corner_pts[30].x, corner_pts[30].y);
    // float d4 = distanceCalculate(corner_pts[30].x, corner_pts[30].y, corner_pts[31].x, corner_pts[31].y);
    // float d5 = distanceCalculate(corner_pts[31].x, corner_pts[31].y, corner_pts[32].x, corner_pts[32].y);
    // float d6 = distanceCalculate(corner_pts[32].x, corner_pts[32].y, corner_pts[23].x, corner_pts[23].y);
    // float d7 = distanceCalculate(corner_pts[23].x, corner_pts[23].y, corner_pts[24].x, corner_pts[24].y);
    // float d8 = distanceCalculate(corner_pts[24].x, corner_pts[24].y, corner_pts[15].x, corner_pts[15].y);
    // float d9 = distanceCalculate(corner_pts[15].x, corner_pts[15].y, corner_pts[16].x, corner_pts[16].y);

    // float avg = (d0+d1+d2+d3+d4+d5+d6+d7+d8+d9)/10;
    // float mse = sqrt(pow(d0-avg,2)
    //                 + pow(d1-avg,2)
    //                 + pow(d2-avg,2)
    //                 + pow(d3-avg,2)
    //                 + pow(d4-avg,2)
    //                 + pow(d5-avg,2)
    //                 + pow(d6-avg,2)
    //                 + pow(d7-avg,2)
    //                 + pow(d8-avg,2)
    //                 + pow(d9-avg,2));

    return mse;

}

void vpi_undist(cv::Mat &img, cv::Size &imgSize, std::vector<double> &coeffs, VPIStream &stream, VPIPayload &remap, VPICameraIntrinsic &K, VPICameraExtrinsic &X, VPIImage &tmpIn, VPIImage &tmpOut, VPIImage &vimg)
{

    // Allocate a dense map.
    VPIWarpMap map = {};
    map.grid.numHorizRegions = 1;
    map.grid.numVertRegions = 1;
    map.grid.regionWidth[0] = imgSize.width;
    map.grid.regionHeight[0] = imgSize.height;
    map.grid.horizInterval[0] = 1;
    map.grid.vertInterval[0] = 1;
    CHECK_STATUS(vpiWarpMapAllocData(&map));

    // Initialize the Polynomial lens model with the coefficients given by calibration procedure.
    VPIPolynomialLensDistortionModel distModel = {};
    distModel.k1 = coeffs[0];
    distModel.k2 = coeffs[1];
    distModel.k3 = coeffs[2];
    distModel.k4 = coeffs[3];

    //? momory 증가 주범 --------------------------------
    vpiWarpMapGenerateFromPolynomialLensDistortionModel(K, X, K, &distModel, &map);
    CHECK_STATUS(vpiCreateRemap(VPI_BACKEND_CUDA, &map, &remap));
    vpiWarpMapFreeData(&map);
    CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CUDA, &stream));
    CHECK_STATUS(vpiImageCreate(imgSize.width, imgSize.height, VPI_IMAGE_FORMAT_NV12_ER, 0, &tmpIn));
    CHECK_STATUS(vpiImageCreate(imgSize.width, imgSize.height, VPI_IMAGE_FORMAT_NV12_ER, 0, &tmpOut));
    //? -----------------------------------------------

    if (vimg == nullptr)
    {
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(img, 0, &vimg));
    }
    else
    {
        CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vimg, img));
    }
    CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, vimg, tmpIn, NULL));
    CHECK_STATUS(vpiSubmitRemap(stream, VPI_BACKEND_CUDA, remap, tmpIn, tmpOut, VPI_INTERP_CATMULL_ROM,
                                VPI_BORDER_ZERO, 0));
    CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, tmpOut, vimg, NULL));
    CHECK_STATUS(vpiStreamSync(stream));
}

 
int main() {

    // Defining the dimensions of checkerboard
    int CHECKERBOARD[2] = {9,5}; 

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
    cv::Mat img, gray, img_und, gray_und;

    // vector to store the pixel coordinates of detected checker board corners 
    vector<cv::Point2f> corner_pts;

    // Looping over all the images in the directory
    img = cv::imread(img_path);

    bool success;
    float mse;
    // ? vpi stream parameter setting
    VPIStream stream = NULL;
    VPIPayload remap = NULL;
    VPIImage tmpIn = NULL, tmpOut = NULL;
    VPIImage vimg = nullptr;
    VPICameraIntrinsic K;
    VPICameraExtrinsic X = {};
    vector<double> coeffs(4);
    Vpi_param stc_vpi_param;

    cv::Mat img_clone;
    cv::Size imgSize;

    if (img.cols != stc_vpi_param.x_resize || img.rows != stc_vpi_param.y_resize)
        cv::resize(img, img, cv::Size(stc_vpi_param.x_resize, stc_vpi_param.y_resize));



    float b_mse = 999.0f;
    for (int i = 0; i < 20; i++) {

        img_clone = img.clone();
        imgSize = img_clone.size();
        stc_vpi_param.vpi_k1 -= 0.00005;

        make_VPICamera_Param(imgSize, stc_vpi_param, K, X, coeffs);
        vpi_undist(img_clone, imgSize, coeffs, stream, remap, K, X, tmpIn, tmpOut, vimg);
        string save_path = "udt_" + to_string(stc_vpi_param.vpi_k1) + "_result.jpg";
        cv::imwrite(save_path, img_clone);

        cv::cvtColor(img_clone, gray_und, cv::COLOR_BGR2GRAY);
        success = cv::findChessboardCorners(gray_und, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (success) {
            mse = make_dist_mse_loss(corner_pts, CHECKERBOARD);
            cout << "[Find Chessboard] [k1 : " << stc_vpi_param.vpi_k1 <<"] | mse : " << mse << endl;

            corner_pts.clear();

            if (b_mse < mse) {
                cout << "b_mse : " << b_mse << " | mse : " << mse << endl;
                break;
            }

            b_mse = mse;
        }
    }

    cout << stc_vpi_param.vpi_k1 << endl;

    return 0;
}