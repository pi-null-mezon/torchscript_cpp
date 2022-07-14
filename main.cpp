#include <iostream>

#include "speechproc.h"

#include <QString>
#include <QFileInfo>
#include <QElapsedTimer>

#ifdef ENABLE_VISUALIZATION

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void showWindowWithPlot(const std::string &_title, const cv::Size _windowsize, const float *_data, const int _datalength, float _ymax, float _ymin, cv::Scalar _color)
{
    if(_datalength > 0 && _windowsize.area() > 0 && _data != NULL ) {

        cv::Mat _colorplot = cv::Mat::zeros(_windowsize, CV_8UC3);
        cv::rectangle(_colorplot,cv::Rect(0,0,_colorplot.cols,_colorplot.rows),cv::Scalar(20,20,20), -1);

        int _ticksX = 10;
        float _tickstepX = static_cast<float>(_windowsize.width)/ _ticksX ;
        for(int i = 1; i < _ticksX ; i++)
            cv::line(_colorplot, cv::Point2f(i*_tickstepX,0), cv::Point2f(i*_tickstepX,static_cast<float>(_colorplot.rows)), cv::Scalar(100,100,100), 1);

        int _ticksY = 10;
        float _tickstepY = static_cast<float>(_windowsize.height)/ _ticksY ;
        for(int i = 1; i < _ticksY ; i++) {
            cv::line(_colorplot, cv::Point2f(0,i*_tickstepY), cv::Point2f(static_cast<float>(_colorplot.cols),i*_tickstepY), cv::Scalar(100,100,100), 1);
            cv::putText(_colorplot, QString::number(_ymax - i * (_ymax-_ymin)/_ticksY,'f',1).toStdString(),
                        cv::Point2f(5, i*_tickstepY - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(150,150,150), 1, cv::LINE_AA);
        }

        float invstepY = static_cast<float>(_ymax - _ymin) / _windowsize.height;
        float stepX = static_cast<float>(_windowsize.width) / (_datalength - 1);

        for(int i = 0; i < _datalength - 1; i++) {
            cv::line(_colorplot, cv::Point2f(i*stepX, _windowsize.height - static_cast<float>(_data[i] - _ymin)/invstepY),
                                 cv::Point2f((i+1)*stepX, _windowsize.height - static_cast<float>(_data[i+1] - _ymin)/invstepY),
                                 _color, 1, cv::LINE_AA);
        }
        cv::imshow(_title, _colorplot);
    }
}
#endif

using namespace std;

int main(int argc, char **argv)
{
    if(argc != 2) {
        std::cout << "Provide filename to read" << std::endl;
        return 1;
    }

    QFileInfo modelfile("./vad.jit");
    if(!modelfile.exists()) {
        std::cerr << QString("model '%1' not found on disk!").arg(modelfile.absoluteFilePath()).toStdString() << std::endl;
        return 1;
    }
    torch::jit::script::Module vad;
    try {
        c10::InferenceMode guard;
        vad = torch::jit::load(modelfile.absoluteFilePath().toStdString());
        vad.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return 2;
    }

    torchaudio::sox_effects::initialize_sox_effects();

    //Processing
    for(int iteration = 0 ; iteration < 5; ++iteration) {
        std::cout << "ITERATION # " << iteration << std::endl;
        auto info = torchaudio::sox_io::get_info_file(argv[1],"wav");
        std::cout << "FILE META INFORMATION" << std::endl;
        std::cout << " - sample rate:         " << std::get<0>(info) << std::endl;
        std::cout << " - samples per channel: " << std::get<1>(info) << std::endl;
        std::cout << " - channels:            " << std::get<2>(info) << std::endl;
        std::cout << " - bits per sample:     " << std::get<3>(info) << std::endl;
        std::cout << " - encoding:            " << std::get<4>(info) << std::endl;

        torch::Tensor tensor = read_audio(argv[1],8000,1);

        QElapsedTimer qet;
        qet.start();
        std::vector<std::pair<int,int>> speech_timestamps = apply_vad_8khz(tensor,vad);
        std::cout << QString::number(qet.elapsed(),'f',1).toStdString() << " ms" << std::endl;

        /*for(const auto &item: speech_timestamps) {
            std::cout << item.first << "-" << item.second << std::endl;
        }*/

        #ifdef ENABLE_VISUALIZATION
        size_t length = tensor.sizes()[1];
        float *audio = new float[length];
        for(size_t i = 0; i < length; ++i)
            audio[i] = tensor[0][i].item().toFloat();
        showWindowWithPlot("probe",cv::Size(1280,480),audio,length,1.0f,-1.0f,cv::Scalar(0,255,0));
        float *speech = new float[length];
        for(size_t i = 0; i < length; ++i)
            speech[i] = 0;
        for(size_t j = 0; j < speech_timestamps.size(); ++j)
            for(int i = speech_timestamps[j].first; i < speech_timestamps[j].second; ++i)
                speech[i] = 1.0f;
        showWindowWithPlot("vad",cv::Size(1280,480),speech,length,1.0f,-1.0f,cv::Scalar(0,255,0));

        cv::waitKey(100);
        #endif
    }
    return 0;
}

