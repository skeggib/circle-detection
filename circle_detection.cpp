#include <iostream>
#include <opencv2/opencv.hpp>
#include "arguments.hpp"
#include <vector>
#include <array>
#include <chrono>
#include <cmath>

struct acc_t : public std::vector<std::vector<std::vector<double>>>
{
    double min_i; double max_i; double step_i;
    double min_j; double max_j; double step_j;
    double min_r; double max_r; double step_r;

    int rows, cols, rays;

    acc_t(
        double min_i, double max_i, double step_i,
        double min_j, double max_j, double step_j,
        double min_r, double max_r, double step_r)
        : std::vector<std::vector<std::vector<double>>>(
            std::ceil((max_i - min_i) / step_i),
            std::vector<std::vector<double>>(
                std::ceil((max_j - min_j) / step_j),
                std::vector<double>(
                    std::ceil((max_r - min_r) / step_r),
                    0)))
    {
        this->min_i = min_i;
        this->max_i = max_i;
        this->step_i = step_i;
        this->min_j = min_j;
        this->max_j = max_j;
        this->step_j = step_j;
        this->min_r = min_r;
        this->max_r = max_r;
        this->step_r = step_r;

        this->rows = std::ceil((max_i - min_i) / step_i);
        this->cols = std::ceil((max_j - min_j) / step_j);
        this->rays = std::ceil((max_r - min_r) / step_r);
    }
};

int main(int argc, char **argv)
{
    arguments_t args(argc, argv);
    if (args.parametersSize() < 3)
    {
        std::cout << "usage: " << argv[0] << " <input> <output> <circles_number>" << std::endl;
        return -1;
    }

    std::string input_path(args[0]);
    std::string output_path(args[1]);
    int circles_number = std::stoi(args[2]);

    auto image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (!image.data)
    {
        std::cout << "Could not open " << input_path << std::endl;
        return -1;
    }

    // Gaussian filter
    cv::Mat blured;
    cv::GaussianBlur(image, blured, cv::Size(9, 9), 2, 2);

    // Sobel filter
    cv::Mat sobelX, absSobelX;
    cv::Sobel(blured, sobelX, CV_16S, 1, 0);
    cv::convertScaleAbs(sobelX, absSobelX);
    cv::Mat sobelY, absSobelY;
    cv::Sobel(blured, sobelY, CV_16S, 0, 1);
    cv::convertScaleAbs(sobelY, absSobelY);
    cv::Mat edges;
    cv::addWeighted(absSobelX, 0.5, absSobelY, 0.5, 0, edges);

    // Creating acc
    acc_t acc(
        0, edges.rows, 1.,
        0, edges.cols, 1.,
        5, std::max(edges.rows - 1, edges.cols - 1) * std::sqrt(2), 1);

    std::cout << acc.rows << "x" << acc.cols << "x" << acc.rays << std::endl;

    std::vector<int> sqrt_table(std::pow(std::max(edges.rows, edges.cols) * std::sqrt(2), 2));
    for (int i = 0; i < sqrt_table.size(); i++)
        sqrt_table[i] = (int)((std::sqrt(i) - acc.min_r) / acc.step_r);

    // Incrementing acc
    auto start = std::chrono::high_resolution_clock::now();
    double min_distance = acc.min_r * acc.min_r;
    auto it_i = acc.begin();
    for (double i = acc.min_i; i < acc.max_i && it_i != acc.end(); i += acc.step_i, it_i++)
    {
        auto it_j = (*it_i).begin();
        for (double j = acc.min_j; j < acc.max_j && it_j != (*it_i).end(); j += acc.step_j, it_j++)
        {
            uchar *pixel = edges.data;
            for (int x = 0; x < edges.cols; x++)
            {
                double j_part = (j - x) * (j - x);
                for (int y = 0; y < edges.rows; y++, pixel++)
                {
                    double i_part = (i - y) * (i - y);
                    double sqr_distance = j_part + i_part;
                    if (sqr_distance >= min_distance)
                        (*it_j)[sqrt_table[sqr_distance]] += (double)*pixel / 255.;
                }
            }
        }
        std::cout << "\rSearching circles... " << (int)(i * 100 / (acc.max_i - acc.min_i)) << "%" << std::flush;
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
                            for (int dr = std::max(-d, -r); dr <= std::min(d, acc.rays - r - 1) && is_local_max; dr++)
                            {
                                double v = acc[i + di][j + dj][r + dr];
                                if (value < v)
                                    is_local_max = false;
                                else if (v < value)
                                    acc[i + di][j + dj][r + dr] = 0;
                            }
                        }
                    }
                    if (!is_local_max)
                        acc[i][j][r] = 0;
                }
            }
        }
        std::cout << "\rSearching local maxima... " << (int)((i + 1) * 100 / acc.rows) << "%" << std::flush;
    }
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << " " << (int)(elapsed.count() * 1000) << " ms" << std::endl;

    // Searching n greater values
    std::vector<double> maxima(circles_number);
    for (int i = 0; i < maxima.size(); i++)
        maxima[i] = 0;
    std::vector<std::array<double, 3>> circles(circles_number);
    for (int i = 0; i < acc.rows; i++)
    {
        for (int j = 0; j < acc.cols; j++)
        {
            for (int r = 0; r < acc.rays; r++)
            {
                for (int l = 0; l < maxima.size(); l++)
                {
                    if (maxima[l] < acc[i][j][r])
                    {
                        for (int m = maxima.size() - 1; m > l; m--)
                        {
                            maxima[m] = maxima[m - 1];
                            circles[m] = circles[m - 1];
                        }
                        maxima[l] = acc[i][j][r];
                        circles[l] =
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

    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_GRAY2RGB);
    for (int i = 0; i < circles.size(); i++)
    {
        std::cout << circles[i][0] << "," << circles[i][1] << " " << circles[i][2] << " " << maxima[i] << std::endl;
        cv::circle(
            result,
            cv::Point(std::round(circles[i][0]), std::round(circles[i][1])),
            std::round(circles[i][2]),
            cv::Scalar(0, 0, 255));
    }

    cv::imwrite(output_path, result);

    // cv::imshow("result", result);
    // cv::waitKey();

    return 0;
}