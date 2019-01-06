#include <iostream>
#include <opencv2/opencv.hpp>
#include "arguments.hpp"
#include <vector>
#include <array>
#include <chrono>
#include <cmath>

struct acc_t : public std::vector<std::vector<std::vector<double>>>
{
    int min_i; int max_i; int step_i;
    int min_j; int max_j; int step_j;
    int min_r; int max_r; int step_r;

    int rows, cols, rays;

    acc_t(
        int min_i, int max_i, int step_i,
        int min_j, int max_j, int step_j,
        int min_r, int max_r, int step_r
    ) : std::vector<std::vector<std::vector<double>>>(
        (max_i - min_i) / step_i + 1,
        std::vector<std::vector<double>>(
            (max_j - min_j) / step_j + 1,
            std::vector<double>(
                (max_r - min_r) / step_r + 1,
                0
            )
        )
    ) {
        this->min_i  = min_i;
        this->max_i  = max_i;
        this->step_i = step_i;
        this->min_j  = min_j;
        this->max_j  = max_j;
        this->step_j = step_j;
        this->min_r  = min_r;
        this->max_r  = max_r;
        this->step_r = step_r;

        this->rows = (max_i - min_i) / step_i;
        this->cols = (max_j - min_j) / step_j;
        this->rays = (max_r - min_r) / step_r;
    }

    void inc(double i, double j, double r, double value)
    {
        this->operator[]((i - min_i) / step_i)[(j - min_j) / step_j][(r - min_r) / step_r] += value;
    }
};

int main(int argc, char** argv)
{
    arguments_t args(argc, argv);
    if (args.parametersSize() < 1)
    {
        std::cout << "usage: " << argv[0] << " <image>" << std::endl;
        return 0;
    }

    auto image = cv::imread(args[0], cv::IMREAD_GRAYSCALE);

    // Gaussian filter
    cv::Mat blured;
    cv::GaussianBlur(image, blured, cv::Size(3, 3), 1);
    
    // Sobel filter
    cv::Mat sobelX, absSobelX;
    cv::Sobel(blured, sobelX, CV_16S, 1, 0);
    cv::convertScaleAbs(sobelX, absSobelX);
    cv::Mat sobelY, absSobelY;
    cv::Sobel(blured, sobelY, CV_16S, 0, 1);
    cv::convertScaleAbs(sobelY, absSobelY);
    cv::Mat edges;
    cv::addWeighted(absSobelX, 0.5, absSobelY, 0.5, 0, edges);

    // Threshold filter
    // double min, max;
    // cv::minMaxLoc(edges, &min, &max);
    // cv::threshold(edges, edges, 0.8 * max, 255, cv::ThresholdTypes::THRESH_BINARY);

    // Creating acc
    acc_t acc(
        0, edges.rows - 1, 1,
        0, edges.cols - 1, 1,
        5, (int)(std::max(edges.rows - 1, edges.cols - 1) * std::sqrt(2)), 1
    );

    // Incrementing acc
    auto start = std::chrono::high_resolution_clock::now();
    for (int x = 0; x < edges.cols; x++)
    {
        for (int y = 0; y < edges.rows; y++)
            for (int i = acc.min_i; i < acc.max_i; i += acc.step_i)
                for (int j = acc.min_j; j < acc.max_j; j += acc.step_j)
                {
                    int distance = (int)std::sqrt((j-x)*(j-x)+(i-y)*(i-y));
                    if (distance >= acc.min_r)
                        acc.inc(i, j, distance, edges.at<uchar>(x, y));
                }
        std::cout << "\rSearching circles... " << (int)((x+1) * 100 / edges.cols) << "%" << std::flush;
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << " " << (int)(elapsed.count() * 1000) << " ms" << std::endl;

    // Normalizing acc
    for (int r = 0; r < acc.rays; r++)
    {
        double circumference = 2 * 3.14 * (r * acc.step_r + acc.min_r);
        for (int i = 0; i < acc.rows; i++)
            for (int j = 0; j < acc.cols; j++)
                acc[i][j][r] /= circumference;
    }
    
    // Finding local maxima
    std::array<std::array<int, 3>, 4> maxCircles;
    std::array<double, 4> maxValues;
    for (int i = 0; i < maxValues.size(); i++)
        maxValues[i] = 0;
    int d = 5;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < acc.rows; i++)
    {
        for (int j = 0; j < acc.cols; j++)
        {
            for (int r = 0; r < acc.rays; r++)
            {
                double value = acc[i][j][r];
                if (value > 0)
                {
                    bool is_local_max = true;
                    for (int di = std::max(-d, -i); di <= std::min(d, acc.cols - i - 1) && is_local_max; di++)
                    {
                        for (int dj = std::max(-d, -j); dj <= std::min(d, acc.rows - j - 1) && is_local_max; dj++)
                        {
                            for (int dr = std::max(-d, -r); dr <= std::min(d, acc.max_r-r-1) && is_local_max; dr++)
                            {
                                double v = acc[i+di][j+dj][r+dr];
                                if (value < v)
                                    is_local_max = false;
                                else if (v < value)
                                    acc[i+di][j+dj][r+dr] = 0;
                            }
                        }
                    }
                    if (!is_local_max)
                        acc[i][j][r] = 0;
                }
            }
        }
        std::cout << "\rSearching local maxima... " << (int)((i+1) * 100 / acc.rows) << "%" << std::flush;
    }
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << " " << (int)(elapsed.count() * 1000) << " ms" << std::endl;
    
    // Searching n greater values
    for (int i = 0; i < acc.rows; i++)
    {
        for (int j = 0; j < acc.cols; j++)
        {
            for (int r = 0; r < acc.rays; r++)
            {
                for (int l = 0; l < maxValues.size(); l++)
                {
                    if (maxValues[l] < acc[i][j][r])
                    {
                        for (int m = l; m < maxValues.size() - 1; m++)
                        {
                            maxValues[m+1] = maxValues[m];
                            maxCircles[m+1] = maxCircles[m];
                        }
                        maxValues[l] = acc[i][j][r];
                        maxCircles[l] = 
                        {
                            i * acc.step_i + acc.min_i,
                            j * acc.step_j + acc.min_j,
                            r * acc.step_r + acc.min_r,
                        };
                        break;
                    }
                }
            }
        }
    }
    
    for(int i = 0; i < maxCircles.size(); i++)
        std::cout << maxCircles[i][0] << "," << maxCircles[i][1] << " " << maxCircles[i][2] << " " << maxValues[i] << std::endl;
    
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_GRAY2RGB);
    for (auto maxIndex : maxCircles)
        cv::circle(result, cv::Point(maxIndex[0], maxIndex[1]), maxIndex[2], cv::Scalar(0, 0, 255));

    cv::imshow("result", result);
    cv::waitKey();

    return 0;
}