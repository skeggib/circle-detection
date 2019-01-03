#include <iostream>
#include <opencv2/opencv.hpp>
#include "arguments.hpp"
#include <vector>
#include <array>
#include <chrono>
#include <cmath>

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

    cv::imshow("edges", edges);

    // Threshold filter
    // double min, max;
    // cv::minMaxLoc(edges, &min, &max);
    // cv::threshold(edges, edges, 0.8 * max, 255, cv::ThresholdTypes::THRESH_BINARY);

    // Creating acc
    std::vector<std::vector<std::vector<double>>> acc;
    int r_max = edges.rows - 1;
    int c_max = edges.cols - 1;
    int rad_min = 5;
    int rad_max = (int)(std::max(r_max, c_max) * std::sqrt(2));
    for (int i = 0; i < r_max; i++)
    {
        std::vector<std::vector<double>> temp1;
        for (int j = 0; j < c_max; j++)
        {
            std::vector<double> temp2;
            for (int k = rad_min; k < rad_max; k++)
                temp2.push_back(0);
            temp1.push_back(temp2);
        }
        acc.push_back(temp1);
    }

    // Incrementing acc
    auto start = std::chrono::high_resolution_clock::now();
    for (int x = 0; x < edges.cols; x++)
    {
        for (int y = 0; y < edges.rows; y++)
            for (int i = 0; i < r_max; i++)
                for (int j = 0; j < c_max; j++)
                {
                    int distance = (int)std::sqrt((j-x)*(j-x)+(i-y)*(i-y));
                    if (distance >= rad_min)
                        acc[i][j][distance - rad_min] += edges.at<uchar>(x, y);
                }
        std::cout << "\rSearching circles... " << (int)((x+1) * 100 / edges.cols) << "%";
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << " " << (int)(elapsed.count() * 1000) << " ms" << std::endl;

    // Normalizing acc
    for (int k = rad_min; k < rad_max; k++)
    {
        double circumference = 2 * 3.14 * k;
        for (int i = 0; i < r_max; i++)
            for (int j = 0; j < c_max; j++)
                acc[i][j][k-rad_min] /= circumference;
    }
    
    // Finding local maxima
    std::array<std::array<int, 3>, 4> maxCircles;
    std::array<double, 4> maxValues;
    for (int i = 0; i < maxValues.size(); i++)
        maxValues[i] = 0;
    int d = 5;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < r_max; i++)
    {
        for (int j = 0; j < c_max; j++)
        {
            for (int k = rad_min; k < rad_max; k++)
            {
                double value = acc[i][j][k - rad_min];
                if (value > 0)
                {
                    bool is_local_max = true;
                    for (int di = std::max(-d, -i); di <= std::min(d, c_max - i - 1) && is_local_max; di++)
                    {
                        for (int dj = std::max(-d, -j); dj <= std::min(d, r_max - j - 1) && is_local_max; dj++)
                        {
                            for (int dk = std::max(-d, -k+rad_min); dk <= std::min(d, rad_max-k-1-rad_min) && is_local_max; dk++)
                            {
                                double v = acc[i+di][j+dj][k+dk-rad_min];
                                if (value < v)
                                    is_local_max = false;
                                else if (v < value)
                                    acc[i+di][j+dj][k+dk-rad_min] = 0;
                            }
                        }
                    }
                    if (!is_local_max)
                        acc[i][j][k-rad_min] = 0;
                }
            }
        }
        std::cout << "\rSearching local maxima... " << (int)((i+1) * 100 / r_max) << "%" << std::flush;
    }
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << " " << (int)(elapsed.count() * 1000) << " ms" << std::endl;
    
    // Searching n greater values
    for (int i = 0; i < r_max; i++)
    {
        for (int j = 0; j < c_max; j++)
        {
            for (int k = rad_min; k < rad_max; k++)
            {
                for (int l = 0; l < maxValues.size(); l++)
                {
                    if (maxValues[l] < acc[i][j][k-rad_min])
                    {
                        for (int m = l; m < maxValues.size() - 1; m++)
                        {
                            maxValues[m+1] = maxValues[m];
                            maxCircles[m+1] = maxCircles[m];
                        }
                        maxValues[l] = acc[i][j][k-rad_min];
                        maxCircles[l] = { i, j, k };
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