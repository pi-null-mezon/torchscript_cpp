# step 1 - download C++ torch binaries from https://pytorch.org
linux {
    old_cxx_abi {
        TORCH_DISTRIB = "/home/alex/Programming/3rdParties/libtorch-shared-with-deps-1.11.0+cpu/libtorch"
        QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
    } else {
        TORCH_DISTRIB = "/home/alex/Programming/3rdParties/libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu/libtorch"
        QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
    }

    INCLUDEPATH += $${TORCH_DISTRIB} \
                   $${TORCH_DISTRIB}/include

    LIBS += -L$${TORCH_DISTRIB}/lib \
            -lc10 \
            -ltorch_cpu   
}
