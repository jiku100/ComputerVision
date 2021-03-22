#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>

//0: 기본, 1: 각도 맞춰준 버전, 2: 똑같은 그림 뒤집은 버전, 3: 동일한 사진
#define MODE 0

#define IMAGE_PATH_1 "1st.jpg"
#define IMAGE_PATH_1R "1st_r.jpg"
#define IMAGE_PATH_1RV "1st_rv.jpg"
#define IMAGE_PATH_2 "2nd.jpg"
#define WINDOW_NAME_1 "1st"
#define WINDOW_NAME_2 "2nd"

#define BIN_NUM 10
#define BLOCK_NUM 4
#define BLOCK_RADIUS 30

#define KERNEL_SIZE 3

#define GUI_LINE_WIDTH 3
#define HIST_BLOCK_SIZE 30
#define HIST_WINDOW_SIZE 200

cv::Mat sobelKerX = (cv::Mat_<char>(1, 3) << -1, 0, 1);
cv::Mat sobelKerY = (cv::Mat_<char>(3, 1) << -1, 0, 1);

cv::Mat srcImg1;
cv::Mat coordImg1;
cv::Mat OoG1;
cv::Mat hist1[BLOCK_NUM];
int blockNum1 = 0;
cv::Point point1[BLOCK_NUM];

cv::Mat srcImg2;
cv::Mat coordImg2;
cv::Mat OoG2;
cv::Mat hist2[BLOCK_NUM];
int blockNum2 = 0;
cv::Point point2[BLOCK_NUM];

int matchList[BLOCK_NUM];

//mode -> 로딩과 실행과정 자동화
void loadImg(int mode);
void autoSelect(int mode);

//calculators -> 전역 변수에 영향
void calHist1(int blockNum);
void calHist2(int blockNum);
void calMatch();

//getting functions -> 중간 연산 과정
cv::Mat getGradX(cv::Mat& img);
cv::Mat getGradY(cv::Mat& img);
cv::Mat getOoG(cv::Mat&);
cv::Mat getHistogram(cv::Mat&);
cv::Mat getBlock(cv::Mat& src, int x, int y);

//graphic functions -> 출력 결과 등을 보여줌
void showHistogram(cv::Mat& histogram, std::string title);
void drawRect(cv::Mat& src, int x, int y);
void drawFont(cv::Mat& src, int x, int y, int blockNum);
void showRes();

//ui functions -> 클릭 이벤트 함수들
void mouseClickEventImg1(int  event, int  x, int  y, int  flag, void* param);
void mouseClickEventImg2(int  event, int  x, int  y, int  flag, void* param);

int main(int, char**)
{
    loadImg(MODE);

    //nullptr detection
    if (srcImg1.empty()) {
        std::cout << IMAGE_PATH_1
            << " 이미지 1을 불러오는 데 문제가 생겼습니다." << std::endl;
        return -1;
    }

    if (srcImg2.empty()) {
        std::cout << IMAGE_PATH_2
            << " 이미지 2를 불러오는 데 문제가 생겼습니다." << std::endl;
        return -1;
    }

    OoG1 = getOoG(srcImg1);
    OoG2 = getOoG(srcImg2);

    //show image
    cv::namedWindow(WINDOW_NAME_1, cv::WINDOW_NORMAL);
    cv::imshow(WINDOW_NAME_1, coordImg1);
    cv::namedWindow(WINDOW_NAME_2, cv::WINDOW_NORMAL);
    cv::imshow(WINDOW_NAME_2, coordImg2);

    //set mouse callback
    /*cv::setMouseCallback(WINDOW_NAME_1, mouseClickEventImg1);
    cv::setMouseCallback(WINDOW_NAME_2, mouseClickEventImg2);*/

    //select points automatically
    autoSelect(MODE);

    //키 입력이 있을 때 까지 기다립니다.
    cv::waitKey(0);

    //생성하였던 윈도우를 제거합니다.
    //cv::destroyWindow(WINDOW_NAME_ORIGINAL);

    //아래의 함수를 사용하면, 사용하고 있던 윈도우 전부를 제거합니다.
    cv::destroyAllWindows();

    return 0;
}




/////////////////////////////////
//getters
cv::Mat getGradX(cv::Mat& img) {
    cv::Mat gx;
    cv::filter2D(img, gx, CV_32F, sobelKerX);

    //Sobel(img, gx, CV_32F, 1, 0, KERNEL_SIZE);

    return gx;
}

cv::Mat getGradY(cv::Mat& img) {
    cv::Mat gy;
    cv::filter2D(img, gy, CV_32F, sobelKerY);

    //Sobel(img, gy, CV_32F, 0, 1, KERNEL_SIZE);

    return gy;
}

cv::Mat getOoG(cv::Mat& img) {
    cv::Mat gr;
    cv::bilateralFilter(img.clone(), img, 10, 50, 50);
    phase(getGradX(img), getGradY(img), gr, true);
    return gr;
}

cv::Mat getHistogram(cv::Mat& img) {
    cv::Mat hist;
    int channels[] = { 0 };
    int dims = 1;
    int histSize = BIN_NUM;
    float angleSize[] = { 0,360 };
    const float* ranges[] = { angleSize };

    calcHist(&img, 1, channels, cv::noArray(), hist, dims, &histSize, ranges);
    return hist;
}

cv::Mat getBlock(cv::Mat& src, int x, int y) {
    return src(cv::Range(y - BLOCK_RADIUS, y + BLOCK_RADIUS), cv::Range(x - BLOCK_RADIUS, x + BLOCK_RADIUS));
}



///////////////////////////////////
//calculators
void calMatch() {
    for (int i = 0; i < 4; i++) {
        double maxVal = std::numeric_limits<double>::min();
        for (int j = 0; j < 4; j++) {
            double correl = cv::compareHist(hist1[i], hist2[j], cv::HISTCMP_CORREL);

            double res = correl;

            if (res > maxVal) {
                maxVal = res;
                matchList[i] = j;
            }
        }
    }
}
void calHist1(int blockNum) {

    int x = point1[blockNum].x;
    int y = point1[blockNum].y;

    cv::Mat cutImage = getBlock(OoG1, x, y);

    drawRect(coordImg1, x, y);
    drawFont(coordImg1, x, y, blockNum);

    hist1[blockNum] = getHistogram(cutImage);
    showHistogram(hist1[blockNum], "1_" + std::to_string(blockNum));

    cv::imshow(WINDOW_NAME_1, coordImg1);
}

void calHist2(int blockNum) {

    int x = point2[blockNum].x;
    int y = point2[blockNum].y;

    cv::Mat cutImage = getBlock(OoG2, x, y);

    drawRect(coordImg2, x, y);
    drawFont(coordImg2, x, y, blockNum);

    hist2[blockNum] = getHistogram(cutImage);
    showHistogram(hist2[blockNum], "2_" + std::to_string(blockNum));

    cv::imshow(WINDOW_NAME_2, coordImg2);
}



//////////////////////////////////////
//mode functions
void loadImg(int mode) {
    //load image
    switch (mode) {
    default:
    case 0:
        srcImg1 = cv::imread(IMAGE_PATH_1, cv::IMREAD_COLOR);
        srcImg2 = cv::imread(IMAGE_PATH_2, cv::IMREAD_COLOR);
        break;
    case 1:
        srcImg1 = cv::imread(IMAGE_PATH_1R, cv::IMREAD_COLOR);
        srcImg2 = cv::imread(IMAGE_PATH_2, cv::IMREAD_COLOR);
        break;
    case 2:
        srcImg1 = cv::imread(IMAGE_PATH_1RV, cv::IMREAD_COLOR);
        srcImg2 = cv::imread(IMAGE_PATH_1, cv::IMREAD_COLOR);
        break;
    case 3:
        srcImg1 = cv::imread(IMAGE_PATH_1, cv::IMREAD_COLOR);
        srcImg2 = cv::imread(IMAGE_PATH_1, cv::IMREAD_COLOR);
    }

    coordImg1 = srcImg1.clone();
    coordImg2 = srcImg2.clone();
    /*cv::cvtColor(srcImg1, srcImg1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(srcImg2, srcImg2, cv::COLOR_BGR2GRAY);*/
}

void autoSelect(int mode) {

    switch (mode) {
    default:
    case 0:
        point1[0].x = 131;
        point1[0].y = 1904;
        point1[1].x = 1735;
        point1[1].y = 3623;
        point1[2].x = 2982;
        point1[2].y = 2414;
        point1[3].x = 1366;
        point1[3].y = 731;
        point2[0].x = 779;
        point2[0].y = 1102;
        point2[1].x = 664;
        point2[1].y = 3560;
        point2[2].x = 2550;
        point2[2].y = 3648;
        point2[3].x = 2568;
        point2[3].y = 1107;
        break;
    case 1:
        point1[0].x = 580;
        point1[0].y = 986;
        point1[1].x = 572;
        point1[1].y = 3338;
        point1[2].x = 2309;
        point1[2].y = 3316;
        point1[3].x = 2290;
        point1[3].y = 985;
        point2[0].x = 779;
        point2[0].y = 1102;
        point2[1].x = 664;
        point2[1].y = 3560;
        point2[2].x = 2550;
        point2[2].y = 3648;
        point2[3].x = 2568;
        point2[3].y = 1107;
        break;
    case 2:
        point1[0].x = 2897;
        point1[0].y = 2125;
        point1[1].x = 1289;
        point1[1].y = 411;
        point1[2].x = 37;
        point1[2].y = 1617;
        point1[3].x = 1655;
        point1[3].y = 3298;
        point2[0].x = 131;
        point2[0].y = 1904;
        point2[1].x = 1735;
        point2[1].y = 3623;
        point2[2].x = 2982;
        point2[2].y = 2414;
        point2[3].x = 1366;
        point2[3].y = 731;
        break;
    case 3:
        point1[0].x = 131;
        point1[0].y = 1904;
        point1[1].x = 1735;
        point1[1].y = 3623;
        point1[2].x = 2982;
        point1[2].y = 2414;
        point1[3].x = 1366;
        point1[3].y = 731;
        point2[0].x = 131;
        point2[0].y = 1904;
        point2[1].x = 1735;
        point2[1].y = 3623;
        point2[2].x = 2982;
        point2[2].y = 2414;
        point2[3].x = 1366;
        point2[3].y = 731;
        break;
    }

    for (int i = 0; i < BLOCK_NUM; i++) {
        calHist1(i);
        calHist2(i);
    }

    showRes();
}


//////////////////////////////////////
//Graphic functions
void drawRect(cv::Mat& src, int x, int y) {
    cv::rectangle(src, cv::Rect(x - BLOCK_RADIUS, y - BLOCK_RADIUS, BLOCK_RADIUS * 2 + 1, BLOCK_RADIUS * 2 + 1), cv::Scalar(255, 0, 0), GUI_LINE_WIDTH);
}
void drawFont(cv::Mat& src, int x, int y, int blockNum) {
    int fontY, fontX;
    if (y > (src.size().height / 2)) {
        fontY = y - BLOCK_RADIUS * 2;
    }
    else {
        fontY = y + BLOCK_RADIUS * 2;
    }
    if (x > (src.size().width / 2)) {
        fontX = x - BLOCK_RADIUS * 2;
    }
    else {
        fontX = x + BLOCK_RADIUS * 2;
    }
    cv::putText(src, std::to_string(blockNum), cv::Point(fontX, fontY), 2, 4.0, cv::Scalar(0, 0, 255), GUI_LINE_WIDTH);
}
void showHistogram(cv::Mat& histogram, std::string title) {
    cv::Mat histImg(HIST_WINDOW_SIZE, HIST_BLOCK_SIZE * BIN_NUM, CV_32F);
    cv::normalize(histogram, histogram, 0, HIST_WINDOW_SIZE, cv::NORM_MINMAX, -1, cv::Mat());
    for (int i = 0; i < BIN_NUM; i++) {
        int x = i * HIST_BLOCK_SIZE;
        int y = HIST_WINDOW_SIZE - histogram.at<float>(i) + 1;
        int height = histogram.at<float>(i);

        cv::rectangle(histImg, cv::Rect(x, y, HIST_BLOCK_SIZE, height), cv::Scalar(255, 255, 255), cv::FILLED);
    }

    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, histImg);

}
void showRes() {
    calMatch();
    cv::Mat resImg;
    cv::hconcat(coordImg1, coordImg2, resImg);

    for (int i = 0; i < BLOCK_NUM; i++) {
        std::cout << i << " -> " << matchList[i] << '\n';
        point2[i].x += coordImg1.size().width;
    }

    for (int i = 0; i < BLOCK_NUM; i++) {
        cv::line(resImg, point1[i], point2[matchList[i]], cv::Scalar(0, 0, 0), 10);
    }

    cv::namedWindow("res", cv::WINDOW_NORMAL);
    cv::imshow("res", resImg);
}


//////////////////////////////////////
//mouse click events for UI
void mouseClickEventImg1(int  event, int  x, int  y, int  flag, void* param) {

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (x >= 0 && y >= 0 && x < srcImg1.size().width && y < srcImg1.size().height) {
            if (blockNum1 == BLOCK_NUM) {
                return;
            }

            calHist1(blockNum1);

            blockNum1++;
            if (blockNum2 == BLOCK_NUM && blockNum1 == BLOCK_NUM) {
                showRes();
            }
        }
    }
}

void mouseClickEventImg2(int  event, int  x, int  y, int  flag, void* param) {

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (x >= 0 && y >= 0 && x < srcImg2.size().width && y < srcImg2.size().height) {
            if (blockNum2 == BLOCK_NUM) {
                return;
            }
            calHist2(blockNum2);

            blockNum2++;
            if (blockNum2 == BLOCK_NUM && blockNum1 == BLOCK_NUM) {
                showRes();
            }
        }
    }
}