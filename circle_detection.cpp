#include <iostream>
#include <opencv2/opencv.hpp>
#include "arguments.hpp"
#include <vector>
#include <array>
#include <chrono>
#include <cmath>

struct accumulator_t
{
    std::vector<std::vector<std::vector<double>>> values;
    int x_min, x_max, x_step;
    int y_min, y_max, y_step;
    int r_min, r_max, r_step;

    accumulator_t(
        int x_min, int x_max, int x_step,
        int y_min, int y_max, int y_step,
        int r_min, int r_max, int r_step
    ) {
        this->x_min = x_min;
        this->x_max = x_max;
        this->x_step = x_step;

        this->y_min = y_min;
        this->y_max = y_max;
        this->y_step = y_step;

        this->r_min = r_min;
        this->r_max = r_max;
        this->r_step = r_step;

        this->values = std::vector<std::vector<std::vector<double>>>(
            (x_max - x_min) / x_step,
            std::vector<std::vector<double>>(
                (y_max - y_min) / y_step,
                std::vector<double>(
                    (r_max - r_min) / r_step,
                    0
                )
            )
        );
    }

    double get(int x, int y, int r) const
    {
        return values[(x - x_min) * x_step][(y - y_min) * y_step][(r - r_min) * r_step];
    }

    void set(int x, int y, int r, double value)
    {
        values[(x - x_min) * x_step][(y - y_min) * y_step][(r - r_min) * r_step] = value;
    }

    void increment(int x, int y, int r, double value = 1)
    {
        values[(x - x_min) * x_step][(y - y_min) * y_step][(r - r_min) * r_step] += value;
    }

    int cols() const
    {
        return (x_max - x_min) / x_step;
    }

    int rows() const
    {
        return (y_max - y_min) / y_step;
    }

    int radiuses() const
    {
        return (r_max - r_min) / r_step;
    }
};

int main(int argc, char** argv)
{
    arguments_t args(argc, argv);
    if (args.parametersSize() < 1)
    {
        std::cout << "usage: " << argv[0] << " <image>" << std::endl;
        return 1;
    }

    auto image = cv::imread(args[0], cv::IMREAD_GRAYSCALE);

    if (!image.data)
    {
        std::cout << "Could not read " << args[0] << std::endl;
        return 1;
    }

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

    int r_max = (int)(std::max(image.cols, image.rows) * std::sqrt(2));
    accumulator_t acc(
        0, image.cols - 1, image.cols / 100,
        0, image.rows - 1, image.rows / 100,
        5, r_max, r_max / 100
    );

    // Incrementing acc
    auto start = std::chrono::high_resolution_clock::now();
    for (int x = 0; x < edges.cols; x++)
    {
        for (int y = 0; y < edges.rows; y++)
            for (int i = acc.y_min; i < acc.y_max; i += acc.y_step)
                for (int j = acc.x_min; j < acc.x_max; j += acc.x_step)
                {
                    int distance = (int)std::sqrt((j-x)*(j-x)+(i-y)*(i-y));
                    acc.increment(j, i, distance, edges.at<uchar>(x, y));
                }
        std::cout << "\rSearching circles... " << (int)((x+1) * 100 / edges.cols) << "%";
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << " " << (int)(elapsed.count() * 1000) << " ms" << std::endl;

    // Normalizing acc
    for (int r = 0; r < acc.radiuses(); r++)
    {
        double circumference = 2 * 3.14 * (r * acc.r_step + acc.r_min);
        for (int x = 0; x < acc.cols(); x++)
            for (int y = 0; y < acc.rows(); y++)
                if (circumference > 0)
                    acc.values[x][y][r] /= circumference;
    }
    
    // Finding local maxima
    std::array<std::array<int, 3>, 4> maxCircles;
    std::array<double, 4> maxValues;
    for (int i = 0; i < maxValues.size(); i++)
        maxValues[i] = 0;
    int d = 5;
    start = std::chrono::high_resolution_clock::now();
    for (int x = 0; x < acc.cols(); x++)
    {
        for (int y = 0; y < acc.rows(); y++)
        {
            for (int r = 0; r < acc.radiuses(); r++)
            {
                double value = acc.values[x][y][r];
                if (value > 0)
                {
                    bool is_local_max = true;
                    for (int dx = std::max(-d, -x); dx <= std::min(d, acc.cols() - x - 1) && is_local_max; dx++)
                    {
                        for (int dy = std::max(-d, -y); dy <= std::min(d, acc.rows() - y - 1) && is_local_max; dy++)
                        {
                            for (int dr = std::max(-d, -r); dr <= std::min(d, acc.radiuses() - r - 1) && is_local_max; dr++)
                            {
                                double v = acc.values[x+dx][y+dy][r+dr];
                                if (value < v)
                                    is_local_max = false;
                                else if (v < value)
                                    acc.values[x+dx][y+dy][r+dr] = 0;
                            }
                        }
                    }
                    if (!is_local_max)
                        acc.values[x][y][r] = 0;
                }
            }
        }
        std::cout << "\rSearching local maxima... " << (int)((x+1) * 100 / acc.cols()) << "%" << std::flush;
    }
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << " " << (int)(elapsed.count() * 1000) << " ms" << std::endl;
    
    // Searching n greater values
    for (int x = 0; x < acc.cols(); x++)
    {
        for (int y = 0; y < acc.rows(); y++)
        {
            for (int r = 0; r < acc.radiuses(); r++)
            {
                for (int l = 0; l < maxValues.size(); l++)
                {
                    if (maxValues[l] < acc.values[x][y][r])
                    {
                        for (int m = l; m < maxValues.size() - 1; m++)
                        {
                            maxValues[m+1] = maxValues[m];
                            maxCircles[m+1] = maxCircles[m];
                        }
                        maxValues[l] = acc.values[x][y][r];
                        maxCircles[l] = 
                        {
                            (x + acc.x_min) * acc.x_step,
                            (y + acc.y_min) * acc.y_step,
                            (r + acc.r_min) * acc.r_step
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