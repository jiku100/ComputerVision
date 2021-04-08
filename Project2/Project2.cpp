#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<Point2f> src1Quad, src2Quad, dstQuad;
Mat src1, src2;

void getTransformImage() {
	Mat H = getPerspectiveTransform(src1Quad, src2Quad);
	Mat dst;
	Mat result;
	warpPerspective(src1, dst, H, src2.size(), INTER_LINEAR);
	Mat ROI = src2(Rect(Point(cvRound(src2Quad[0].x), cvRound(src2Quad[0].y)), Point(cvRound(src2Quad[2].x), cvRound(src2Quad[2].y))));
	Mat dstROI = dst(Rect(Point(cvRound(src2Quad[0].x), cvRound(src2Quad[0].y)), Point(cvRound(src2Quad[2].x), cvRound(src2Quad[2].y))));
	dstROI.copyTo(ROI);
	imshow("result", src2);
}

void on_mouse_src1(int event, int x, int y, int flags, void* userdata) {
	static int cnt = 0;
	if (event == EVENT_LBUTTONDOWN) {
		if (cnt < 4) {
			src1Quad.push_back(Point2f(x, y));
			cnt++;
			circle(src1, Point(x, y), 5, Scalar(0, 0, 255), -1);
			imshow("src1", src1);
		}
	}
}

void on_mouse_src2(int event, int x, int y, int flags, void* userdata) {
	static int cnt = 0;
	if (event == EVENT_LBUTTONDOWN) {
		if (cnt < 4) {
			src2Quad.push_back(Point2f(x, y));
			cnt++;
			circle(src2, Point(x, y), 5, Scalar(0, 0, 255), -1);
			imshow("src2", src2);
		}
		else {
			getTransformImage();
		}
	}
}

bool isFourClick() {
	if (src1Quad.size() == 4 && src2Quad.size() == 4)
		return true;
	return false;
}



int main(void) {
	src1 = imread("./K.jpg");
	src2 = imread("./BackGround.jpg");
	namedWindow("src1");
	namedWindow("src2");
	setMouseCallback("src1", on_mouse_src1);
	setMouseCallback("src2", on_mouse_src2);
	imshow("src1", src1);
	imshow("src2", src2); 
	waitKey(0);

	while (!isFourClick()) {};
	getTransformImage();
	waitKey(0);
	destroyAllWindows();

}