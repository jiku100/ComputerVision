#include "opencv2/opencv.hpp"
#include <vector>
#include <math.h>
using namespace cv;
using namespace std;


Mat src1;
Mat src2;
Mat src1_clone;
Mat src2_clone;
vector<Point> pts_1;
vector<Point> pts_2;

void mouse_src1(int event, int x, int y, int flags, void*) {
	static int count = 0;
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		if (count < 4) {
			count++;
			Point p = Point(x, y);
			cout << p << endl;
			/*String order = format("%d", count);
			putText(src1_clone, order, p + Point(-5, -5), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 2);
			rectangle(src1_clone, Rect(p.x - 3, p.y - 3, 6, 6), Scalar(0, 0, 0), 1);
			pts_1.push_back(p);
			imshow("src1", src1_clone);*/
		}
		break;
	default:
		break;
	}
}
void mouse_src2(int event, int x, int y, int flags, void*) {
	static int count = 0;
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		if (count < 4) {
			count++;
			Point p = Point(x, y);
			cout << p << endl;

			String order = format("%d", count);
			putText(src2_clone, order, p + Point(-5, -5), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
			rectangle(src2_clone, Rect(p.x - 3, p.y - 3, 6, 6), Scalar(0, 0, 0), 1);
			pts_2.push_back(p);
			imshow("src2", src2_clone);
		}
		break;
	default:
		break;
	}
}

Mat calcAngleHist(const Mat& img) {
	Mat hist;
	int channels[] = { 0 };
	int dims = 1;
	const int histSize[] = { 8 };
	float angleSize[] = { 0,360 };
	const float* ranges[] = { angleSize };
	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges);
	return hist;
}

Mat getAngleHistImg(const Mat& hist) {
	Mat imgHist(300, 360, CV_8UC1, Scalar(255));
	for (int i = 0; i < 8; i++) {
		int value = hist.at<float>(i, 0);
		line(imgHist, Point(i * 45, 300), Point(i * 45, 300 - value), Scalar(0));
		line(imgHist, Point(i * 45, 300 - value), Point((i + 1) * 45, 300 - value), Scalar(0));
		line(imgHist, Point((i + 1) * 45, 300 - value), Point((i + 1) * 45, 300), Scalar(0));
	}
	return imgHist;
}

int main(void) {
	src1 = imread("1st.jpg", IMREAD_GRAYSCALE);
	resize(src1, src1, Size(416, 416), 0, 0, INTER_CUBIC);
	cvtColor(src1, src1_clone, COLOR_GRAY2BGR);
	src2 = imread("2nd.jpg", IMREAD_GRAYSCALE);
	resize(src2, src2, Size(416, 416), 0, 0, INTER_CUBIC);
	cvtColor(src2, src2_clone, COLOR_GRAY2BGR);
	pts_1.push_back(Point(188, 80)); pts_1.push_back(Point(408, 248)); pts_1.push_back(Point(238, 372)); pts_1.push_back(Point(19, 196));
	for (int i = 0; i < pts_1.size(); i++) {
		String order = format("%d", i + 1);
		putText(src1_clone, order, pts_1[i] + Point(-5, -5), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 6);
		rectangle(src1_clone, Rect(pts_1[i].x - 5, pts_1[i].y - 5, 10, 10), Scalar(0, 0, 0), 1);
	}
	/*pts_2.push_back(Point(1366, 734)); pts_2.push_back(Point(2980, 2414)); pts_2.push_back(Point(1733, 3614)); pts_2.push_back(Point(135, 1905));
	for (int i = 0; i < pts_2.size(); i++) {
		String order = format("%d", i + 1);
		putText(src2_clone, order, pts_2[i] + Point(-5, -5), FONT_HERSHEY_SIMPLEX, 15, Scalar(0, 0, 255), 6);
		rectangle(src2_clone, Rect(pts_2[i].x - 16, pts_2[i].y - 16, 32, 32), Scalar(0, 0, 0), 10);
	}*/
	pts_2.push_back(Point(353, 115)); pts_2.push_back(Point(351, 375)); pts_2.push_back(Point(91, 367)); pts_2.push_back(Point(109, 116));
	for (int i = 0; i < pts_1.size(); i++) {
		String order = format("%d", i + 1);
		putText(src2_clone, order, pts_2[i] + Point(-5, -5), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 6);
		rectangle(src2_clone, Rect(pts_2[i].x - 5, pts_2[i].y - 5, 10, 10), Scalar(0, 0, 0), 1);
	}


	vector<Mat> s1_hist;
	vector<Mat> s2_hist;
	int i = 1;
	for (Point p : pts_1) {
		Mat roi = src1(Rect(p.x - 4, p.y - 4, 8, 8)).clone();
		Mat dx, dy;
		Scharr(roi, dx, CV_32F, 1, 0);
		Scharr(roi, dy, CV_32F, 0, 1);
		Mat angle;
		phase(dx, dy, angle, true);
		Mat hist = calcAngleHist(angle);
		s1_hist.push_back(hist);
		Mat histimg = getAngleHistImg(hist);
		String winname = format("src1_%d", i++);
		imshow(winname, histimg);
	}
	i = 1;
	for (Point p : pts_2) {
		Mat roi = src2(Rect(p.x - 4, p.y - 4, 8, 8)).clone();
		Mat dx, dy;
		Scharr(roi, dx, CV_32F, 1, 0);
		Scharr(roi, dy, CV_32F, 0, 1);
		Mat angle;
		phase(dx, dy, angle, true);
		Mat hist = calcAngleHist(angle);
		s2_hist.push_back(hist);
		Mat histimg = getAngleHistImg(hist);
		String winname = format("src2_%d", i++);
		imshow(winname, histimg);
	}

	Mat output = Mat::zeros(Size(src1_clone.cols * 2, src2_clone.rows), CV_8UC3);
	src1_clone.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
	src2_clone.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

	/*for (int j = 0; j < s1_hist.size(); j++) {
		Mat s1 = s1_hist[j];
		float score = 0;
		float value = 900;
		int idx = 0;
		for (int i = 0; i < s2_hist.size(); i++) {
			score = 0;
			for (int c = 0; c < 8; c++) {
				score += abs(s1.at<float>(c, 0) - s2_hist[i].at<float>(c, 0));
			}
			cout << score << endl;
			if (score < value) {
				value = score;
				idx = i;
			}
		}
		cout << endl;
		line(output, pts_1[j], pts_2[idx] + Point(src1.cols, 0), Scalar(0, 0, 0), 2);
	}*/

	for (int j = 0; j < s1_hist.size(); j++) {
		Mat s1 = s1_hist[j];
		float score = 0;
		float value = 0;
		int idx = 0;
		for (int i = 0; i < s2_hist.size(); i++) {
			score = compareHist(s1, s2_hist[i], HISTCMP_CORREL);
			
			if (score > value) {
				value = score;
				idx = i;
			}
		}
		line(output, pts_1[j], pts_2[idx] + Point(src1.cols, 0), Scalar(0, 0, 0), 1);
	}

	resize(output, output, Size(416 * 2, 416), 0, 0, INTER_CUBIC);
	imshow("result", output);
	waitKey(0);
	destroyAllWindows();
}