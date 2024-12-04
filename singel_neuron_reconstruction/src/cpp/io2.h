#ifndef _IO2_
#define _IO2_

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <QtGui>
#include <QtCore>
#include <QDateTime>
#include <QApplication>
#include <unordered_set>
#include <unordered_map>

#include "v3d_interface.h"
#include "basic_surf_objs.h"
#include "data_definition.h"

using namespace std;

bool getLandmarkTeraFly1(V3DPluginCallback2& callback,V3DPluginArgList &input, V3DPluginArgList &output);
bool saveMarker_file1(const char* marker_file, vector<MyMarker *> &outmarkers);    
bool saveSWCFile(const string savefile, NodeList& neuronTree, bool verbose= false);
V3DLONG* getDimTeraFly1(V3DPluginCallback2& callback,QString &input);
bool getSubVolumeFromTeraFly1(V3DPluginCallback2 &callback, char *imagePath, Image4DSimple &subVolumeImage, 
                            V3DLONG xb, V3DLONG xe, V3DLONG yb, V3DLONG ye, V3DLONG zb, V3DLONG ze,V3DLONG *originSize);
bool normalization(Image4DSimple *image);
bool readSWCtoNodeList(const string filePath, NodeList &nt);
void file_copy(string src, string target);
#endif //_IO2_