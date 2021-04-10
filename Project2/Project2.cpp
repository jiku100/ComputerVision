#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// left top, left bottom, right bottom, right top 순서대로 클릭해야 함

// OPENCV 함수로 구현: 1, 직접 구현: 0

#define OPENCV_VERSION 1

#define IMG_SRC "K.jpg"
#define IMG_DST "BackGround.jpg"
#define WINDOW_NAME_SRC "src"
#define WINDOW_NAME_DST "dst"
#define WINDOW_NAME_RES "result"

std::vector<cv::Point2f> srcQuad, dstQuad;
cv::Mat src, dst, res, H;

int srcClickNum = 0, dstClickNum = 0;

// 공통 함수: 마우스 클릭

void onMouseClick1(int event, int x, int y, int flags, void* userdata);

void onMouseClick2(int event, int x, int y, int flags, void* userdata);

// OPENCV version 함수
void opencv_warp();

// 직접 구현 version의 함수
cv::Mat getHMat();
void warp();

int main(void) {
    src = cv::imread(IMG_SRC);
    dst = cv::imread(IMG_DST);
    res = dst.clone();
    cv::namedWindow(WINDOW_NAME_SRC);
    cv::namedWindow(WINDOW_NAME_DST);
    cv::setMouseCallback(WINDOW_NAME_SRC, onMouseClick1);
    cv::setMouseCallback(WINDOW_NAME_DST, onMouseClick2);
    cv::imshow(WINDOW_NAME_SRC, src);
    cv::imshow(WINDOW_NAME_DST, dst);
    cv::waitKey(0);
}



////////////////////
//Using OpenCV
////////////////////
void opencv_warp() {
    cv::Mat H = cv::getPerspectiveTransform(srcQuad, dstQuad);
    cv::Mat temp;
    cv::Mat result;
    warpPerspective(src, temp, H, dst.size(), cv::INTER_LINEAR);
    cv::Mat mask = cv::Mat::zeros(dst.size(), CV_8UC1);
    std::vector<cv::Point> tmp_point;
    for (cv::Point2f p : dstQuad) {
        tmp_point.push_back(cv::Point(cvRound(p.x), cvRound(p.y)));
    }
    cv::fillConvexPoly(mask, tmp_point, cv::Scalar(255));
    temp.copyTo(res, mask);
  
    cv::namedWindow(WINDOW_NAME_RES);
    cv::imshow(WINDOW_NAME_RES, res);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

////////////////////
//Getter functions
////////////////////
cv::Mat getHMat() {
    cv::Mat A, W, U, VT, V;
    int x1 = srcQuad[0].x;
    int y1 = srcQuad[0].y;
    int x2 = srcQuad[1].x;
    int y2 = srcQuad[1].y;
    int x3 = srcQuad[2].x;
    int y3 = srcQuad[2].y;
    int x4 = srcQuad[3].x;
    int y4 = srcQuad[3].y;

    int xp1 = dstQuad[0].x;
    int yp1 = dstQuad[0].y;
    int xp2 = dstQuad[1].x;
    int yp2 = dstQuad[1].y;
    int xp3 = dstQuad[2].x;
    int yp3 = dstQuad[2].y;
    int xp4 = dstQuad[3].x;
    int yp4 = dstQuad[3].y;

    A = (cv::Mat_<float>(8, 9) <<
        -x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1,
        0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1,
        -x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2,
        0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2,
        -x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3,
        0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3,
        -x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4,
        0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4);

    cv::SVDecomp(A, W, U, VT, cv::SVD::FULL_UV);
    V = VT.t();

    cv::Mat col_vector = V.col(V.cols - 1);
    col_vector = col_vector / col_vector.at<float>(V.rows - 1, 0);
    cv::Mat H(cv::Size(3, 3), CV_32FC1);

    H.at<float>(0, 0) = col_vector.at<float>(0, 0);
    H.at<float>(0, 1) = col_vector.at<float>(1, 0);
    H.at<float>(0, 2) = col_vector.at<float>(2, 0);
    H.at<float>(1, 0) = col_vector.at<float>(3, 0);
    H.at<float>(1, 1) = col_vector.at<float>(4, 0);
    H.at<float>(1, 2) = col_vector.at<float>(5, 0);
    H.at<float>(2, 0) = col_vector.at<float>(6, 0);
    H.at<float>(2, 1) = col_vector.at<float>(7, 0);
    H.at<float>(2, 2) = col_vector.at<float>(8, 0);

    return H;
}

/**해당 점이 내 선택범위 내에 있는지 판별*/
bool isInside(cv::Point2f targetPoint) {
    int crosses = 0;
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;
        if ((srcQuad[i].y > targetPoint.y) != (srcQuad[j].y > targetPoint.y)) {
            float atX = (srcQuad[j].x - srcQuad[i].x) * (targetPoint.y - srcQuad[i].y) / (srcQuad[j].y - srcQuad[i].y) + srcQuad[i].x;
            if (targetPoint.x < atX)
                crosses++;
        }
    }
    return crosses % 2 > 0;
}


////////////////////
//Result Window
////////////////////
void warp() {
    H = getHMat().clone();
    cv::Mat H_inv = H.inv();

    for (int y = 0; y < res.rows; y++) {
        for (int x = 0; x < res.cols; x++) {
            cv::Point3f coor(x, y, 1);
            cv::Mat tmp_coor = (H_inv * cv::Mat(coor));
            tmp_coor /= tmp_coor.at<float>(2, 0);

            cv::Point trans_coor((cvRound(tmp_coor.at<float>(0, 0))), (cvRound(tmp_coor.at<float>(1, 0))));
            float tx = cvRound(trans_coor.x);
            float ty = cvRound(trans_coor.y);
            float a = trans_coor.x - tx;
            float b = trans_coor.y - ty;

            if (isInside(cv::Point2f(tx, ty))) {
                for (int i = 0; i < 3; i++) {
                    res.at<cv::Vec3b>(y, x)[i] = cvRound(
                        (1.0 - a) * (1.0 - b) * src.at<cv::Vec3b>(ty, tx)[i] +
                        a * (1.0 - b) * src.at<cv::Vec3b>(ty, tx + 1)[i] +
                        a * b * src.at<cv::Vec3b>(ty + 1, tx + 1)[i] +
                        (1.0 - a) * b * src.at<cv::Vec3b>(ty + 1, tx)[i]);
                }
            }
        }
    }

    cv::namedWindow(WINDOW_NAME_RES);
    imshow(WINDOW_NAME_RES, res);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

////////////////////
//UI functions
////////////////////
void onMouseClick1(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (srcClickNum < 4) {
            srcQuad.push_back(cv::Point2f(x, y));
            srcClickNum++;
            circle(src, cv::Point2f(x, y), 5, cv::Scalar(0, 0, 255), -1);
            imshow(WINDOW_NAME_SRC, src);
            if (srcClickNum == 4 && dstClickNum == 4) {
#if OPENCV_VERSION == 1
                opencv_warp();
#else
                warp();
#endif
            }
        }
    }
}

void onMouseClick2(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (dstClickNum < 4) {
            dstQuad.push_back(cv::Point2f(x, y));
            dstClickNum++;
            circle(dst, cv::Point2f(x, y), 5, cv::Scalar(0, 0, 255), -1);
            imshow(WINDOW_NAME_DST, dst);
            if (srcClickNum == 4 && dstClickNum == 4) {
#if OPENCV_VERSION == 1
                opencv_warp();
#else
                warp();
#endif
            }
        }
    }
}