TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle

SOURCES += \
        main.cpp \
        speechproc.cpp

include(sndfile.pri)
include(torchaudio.pri)
include(torch.pri)

enable_visualization {
    include(opencv.pri)
    DEFINES+=ENABLE_VISUALIZATION
}

HEADERS += \
    speechproc.h

