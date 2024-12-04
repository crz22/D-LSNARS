
TEMPLATE	= lib
CONFIG	+= qt plugin warn_off
#CONFIG	+= x86_64
VAA3DPATH = D:/ruanjian/Vaa3D/v3d_external
INCLUDEPATH	+= $$VAA3DPATH/v3d_main/basic_c_fun

HEADERS	+= singel_neuron_reconstruction_plugin.h
SOURCES	+= singel_neuron_reconstruction_plugin.cpp
SOURCES	+= $$VAA3DPATH/v3d_main/basic_c_fun/v3d_message.cpp

TARGET	= $$qtLibraryTarget(singel_neuron_reconstruction)
DESTDIR	= $$VAA3DPATH/bin/plugins/singel_neuron_reconstruction/
