#include "arguments.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    arguments_t args(argc, argv);
    if (args.parametersSize() < 1)
    {
        std::cout << "usage: " << argv[0] << " <image>" << std::endl;
        return 0;
    }

    std::string imagePath(args[0]);

    auto image = cv::imread(args[0], cv::IMREAD_GRAYSCALE);

    // Gaussian filter
    cv::Mat blured;
    cv::GaussianBlur(image, blured, cv::Size(3, 3), 1);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blured, circles, cv::HOUGH_GRADIENT, 1, blured.rows / 8, 200, 10);

    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_GRAY2RGB);
    for (int i = 0; i < circles.size(); i++)
    {
        std::cout << circles[i][0] << " " << circles[i][1] << " " << circles[i][2] << std::endl;
        cv::circle(result, cv::Point(circles[i][0], circles[i][1]), circles[i][2], cv::Scalar(0, 0, 255));
    }

    cv::imshow("result", result);
    cv::waitKey();

    return 0;
}