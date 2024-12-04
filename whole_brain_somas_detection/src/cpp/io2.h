#ifndef __IO2_H__
#define __IO2_H__

#include <io.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "v3d_interface.h"
#include "v3d_message.h"
using namespace std;


void getAllFiles(string path, vector<string>& files,string fileType) ;
bool getVolume(V3DPluginCallback2 &callback, char *imagePath, Image4DSimple &VolumeImage, V3DLONG xb=0, V3DLONG xe=256,
                  V3DLONG yb=0, V3DLONG ye=256, V3DLONG zb=0, V3DLONG ze=256);
bool getVolume1(V3DPluginCallback2 &callback, char *imagePath, Image4DSimple &VolumeImage, V3DLONG xb=0, V3DLONG xe=256,
                  V3DLONG yb=0, V3DLONG ye=256, V3DLONG zb=0, V3DLONG ze=256);
bool normalization(Image4DSimple *image);
bool MPL(Image4DSimple *image);


#endif //__IO2_H__