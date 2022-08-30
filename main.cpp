#include <iostream>

#include "speechproc.h"

#include <QString>
#include <QFileInfo>
#include <QElapsedTimer>

#ifdef ENABLE_VISUALIZATION

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void showWindowWithPlot(const std::string &_title, const cv::Size _windowsize, const float *_data, const int _datalength, float _ymax, float _ymin, cv::Scalar _color, cv::Mat &_colorplot, bool show=true)
{
    if(_datalength > 0 && _windowsize.area() > 0 && _data != NULL ) {

        if(_colorplot.empty()) {
            _colorplot = cv::Mat::zeros(_windowsize, CV_8UC3);
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
        }

        float invstepY = static_cast<float>(_ymax - _ymin) / _windowsize.height;
        float stepX = static_cast<float>(_windowsize.width) / (_datalength - 1);

        for(int i = 0; i < _datalength - 1; i++) {
            cv::line(_colorplot, cv::Point2f(i*stepX, _windowsize.height - static_cast<float>(_data[i] - _ymin)/invstepY),
                                 cv::Point2f((i+1)*stepX, _windowsize.height - static_cast<float>(_data[i+1] - _ymin)/invstepY),
                                 _color, 1, cv::LINE_AA);
        }
        if(show)
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

    // INITIALIZATION
    torchaudio::sox_effects::initialize_sox_effects();

    // voice activity detector
    QFileInfo modelfile("./silero_vad_net.jit");
    if(!modelfile.exists()) {
        std::cerr << QString("model '%1' not found on disk!").arg(modelfile.absoluteFilePath()).toStdString() << std::endl;
        return 1;
    }
    torch::jit::script::Module vad_silero;
    try {
        c10::InferenceMode guard;
        vad_silero = torch::jit::load(modelfile.absoluteFilePath().toStdString());
        vad_silero.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading silero VAD model\n";
        return 2;
    }
    // voice language classifier
    modelfile.setFile("./silero_lang_net.jit");
    if(!modelfile.exists()) {
        std::cerr << QString("model '%1' not found on disk!").arg(modelfile.absoluteFilePath()).toStdString() << std::endl;
        return 1;
    }
    torch::jit::script::Module lang_silero;
    try {
        c10::InferenceMode guard;
        lang_silero = torch::jit::load(modelfile.absoluteFilePath().toStdString());
        lang_silero.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading SILERO LANG model\n";
        return 2;
    }
    // sequence predictor
    modelfile.setFile("./sequence_net.jit");
    if(!modelfile.exists()) {
        std::cerr << QString("model '%1' not found on disk!").arg(modelfile.absoluteFilePath()).toStdString() << std::endl;
        return 1;
    }
    torch::jit::script::Module sequence_model;
    try {
        c10::InferenceMode guard;
        sequence_model = torch::jit::load(modelfile.absoluteFilePath().toStdString());
        sequence_model.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading SEQUENCE model\n";
        return 2;
    }
    // many voices model
    modelfile.setFile("./singlevoice_net.jit");
    if(!modelfile.exists()) {
        std::cerr << QString("model '%1' not found on disk!").arg(modelfile.absoluteFilePath()).toStdString() << std::endl;
        return 1;
    }
    torch::jit::script::Module singlevoice_model;
    try {
        c10::InferenceMode guard;
        singlevoice_model = torch::jit::load(modelfile.absoluteFilePath().toStdString());
        singlevoice_model.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading VOICES model\n";
        return 2;
    }

    // biolut voice activity detector
    modelfile.setFile("./bisolut_vad_net.jit");
    if(!modelfile.exists()) {
        std::cerr << QString("model '%1' not found on disk!").arg(modelfile.absoluteFilePath()).toStdString() << std::endl;
        return 1;
    }
    torch::jit::script::Module vad_bisolut;
    try {
        c10::InferenceMode guard;
        vad_bisolut = torch::jit::load(modelfile.absoluteFilePath().toStdString());
        vad_bisolut.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading bisolut VAD model\n";
        return 2;
    }

    // biolut LANG activity detector
    modelfile.setFile("./bisolut_lang_net.jit");
    if(!modelfile.exists()) {
        std::cerr << QString("model '%1' not found on disk!").arg(modelfile.absoluteFilePath()).toStdString() << std::endl;
        return 1;
    }
    torch::jit::script::Module lang_bisolut;
    try {
        c10::InferenceMode guard;
        lang_bisolut = torch::jit::load(modelfile.absoluteFilePath().toStdString());
        lang_bisolut.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading bisolut LANG model\n";
        return 2;
    }

    // PROCESSING
    for(int iteration = 0 ; iteration < 4; ++iteration) {
        std::cout << "ITERATION # " << iteration << "\n--------------" <<std::endl;

        std::cout << "FILE META INFORMATION (libsndfile)" << std::endl;
        SF_INFO sfinfo;
        int bit_depth;
        std::string format_name;

        QFile wavfile(argv[1]);
        wavfile.open(QIODevice::ReadOnly);
        QByteArray wav_bin_data = wavfile.readAll();
        //const uint8_t *content, uint64_t content_size, int target_sampling_rate, SF_INFO &sfinfo
        torch::Tensor wav8 = read_audio_sndfile((uint8_t*)wav_bin_data.data(),wav_bin_data.size(),8000,sfinfo);

        //torch::Tensor wav8 = read_audio_sndfile(argv[1],8000,sfinfo);
        check_sf_format(sfinfo,format_name,bit_depth);
        std::cout << " - sample rate:         " << sfinfo.samplerate << std::endl;
        std::cout << " - samples per channel: " << sfinfo.frames << std::endl;
        std::cout << " - channels:            " << sfinfo.channels << std::endl;
        std::cout << " - bits per sample:     " << bit_depth << std::endl;
        std::cout << " - encoding:            " << format_name << std::endl;

        auto info = torchaudio::sox_io::get_info_file(argv[1],"wav");
        std::cout << "FILE META INFORMATION (libsox)" << std::endl;
        std::cout << " - sample rate:         " << std::get<0>(info) << std::endl;
        std::cout << " - samples per channel: " << std::get<1>(info) << std::endl;
        std::cout << " - channels:            " << std::get<2>(info) << std::endl;
        std::cout << " - bits per sample:     " << std::get<3>(info) << std::endl;
        std::cout << " - encoding:            " << std::get<4>(info) << std::endl;


        //wav8 = read_audio_torchaudio(argv[1],8000,1);
        torch::Tensor wav16 = read_audio_torchaudio(argv[1],16000,1);

        QElapsedTimer qet;
        qet.start();
        std::vector<std::pair<int,int>> silero_speech_timestamps = apply_silero_vad_8khz(wav8,vad_silero);
        std::cout << "vad_silero duration: " <<  QString::number(qet.elapsed(),'f',1).toStdString() << " ms" << std::endl;
        /*for(const auto &item: speech_timestamps) {
            std::cout << item.first << "-" << item.second << std::endl;
        }*/

        qet.start();
        std::vector<std::pair<int,int>> bisolut_speech_timestamps = apply_bisolut_vad_8khz(wav8,vad_bisolut);
        std::cout << "vad_bisolut duration: " <<  QString::number(qet.elapsed(),'f',1).toStdString() << " ms" << std::endl;

        qet.start();
        std::cout << " - russian language prob (SILERO): " << russian_language_prob(wav16,lang_silero) << std::endl;
        std::cout << "SILERO LANG duration: " <<  QString::number(qet.elapsed(),'f',1).toStdString() << " ms" << std::endl;
        qet.start();
        std::cout << " - russian language prob (BISOLUT): " << ru_prob(wav8,lang_bisolut) << std::endl;
        std::cout << "BISOLUT LANG duration: " <<  QString::number(qet.elapsed(),'f',1).toStdString() << " ms" << std::endl;

        std::cout << " - overload: " << estimate_overload(wav8,8000) << std::endl;
        std::cout << " - upsampled: " << estimate_energy_below_frequency(wav16,16000,4000.0f) << std::endl;
        std::cout << " - record duration: " << record_duration(wav8,8000) << " s" << std::endl;

        for(const auto & speech_timestamps : {silero_speech_timestamps, bisolut_speech_timestamps}) {
            std::cout << "   - speech duration: " << speech_duration(speech_timestamps,8000) << " s" << std::endl;
            std::cout << "   - snr: " <<  estimate_snr(wav8,speech_timestamps,8000) << " dB" << std::endl;
            qet.start();
            const std::vector<std::string> sequence = predict_sequence(wav8,speech_timestamps,sequence_model);
            std::cout << "  SEQUENCE duration: " <<  QString::number(qet.elapsed(),'f',1).toStdString() << " ms" << std::endl;
            std::cout << "   - sequence: '";
            for(const auto & item: sequence)
                std::cout << item;
            std::cout << "'" << std::endl;
        }

        qet.start();
        std::cout << " - many voices prob: " << many_voices_prob(wav8,singlevoice_model) << std::endl;
        std::cout << "MANY VOICES duration: " <<  QString::number(qet.elapsed(),'f',1).toStdString() << " ms" << std::endl;

        #ifdef ENABLE_VISUALIZATION
        cv::Mat plot;
        const cv::Size plot_size(2000,400);
        size_t length = wav8.sizes()[1];
        float *audio = new float[length];
        for(size_t i = 0; i < length; ++i)
            audio[i] = wav8[0][i].item().toFloat();
        showWindowWithPlot("probe",plot_size,audio,length,1.0f,-1.0f,cv::Scalar(0,255,0),plot,false);
        delete [] audio;

        float *speech = new float[length];
        for(size_t i = 0; i < length; ++i)
            speech[i] = 0;
        for(size_t j = 0; j < silero_speech_timestamps.size(); ++j)
            for(int i = silero_speech_timestamps[j].first; i < silero_speech_timestamps[j].second; ++i)
                speech[i] = 1.0f;
        showWindowWithPlot("probe",plot_size,speech,length,1.0f,-1.0f,cv::Scalar(0,100,255),plot,false);
        delete [] speech;

        speech = new float[length];
        for(size_t i = 0; i < length; ++i)
            speech[i] = 0;
        for(size_t j = 0; j < bisolut_speech_timestamps.size(); ++j)
            for(int i = bisolut_speech_timestamps[j].first; i < bisolut_speech_timestamps[j].second; ++i)
                speech[i] = 0.5f;
        showWindowWithPlot("probe",plot_size,speech,length,1.0f,-1.0f,cv::Scalar(255,100,0),plot);
        delete [] speech;

        cv::waitKey(0);
        #endif
    }
    return 0;
}

