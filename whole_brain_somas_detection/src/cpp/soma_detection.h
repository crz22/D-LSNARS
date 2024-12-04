#ifndef _SOMA_DETECTION_
#define _SOMA_DETECTION_

#include <iostream>
#include <fstream>
#include <string>
#include <direct.h>
//#include <python>
#include <QtGui>
#include <QtCore/QVariant>
#include <QDateTime>
#include "v3d_interface.h"
#include "basic_surf_objs.h"
#include "io2.h"


class Soma_Detection: public QObject
{
    Q_OBJECT

public:
    QString imagePath;
    QString outputFolderPath;
    QString pytorchPath;
    float CE_value, SMA_value, DB_eps_value, DB_MP_value;

    bool terafly = true;
    vector<string> imagepathlist;
    //BlockSimpleList blockList, allTargetList;

private:
    V3DPluginCallback2& callback;
    QString tempFolderPath;
    QString configurationPath;
    string codePath;
    vector<string> parasString;
    V3DLONG* imageSize = new V3DLONG[4]();
    int batchsize = 256;

public:
    explicit Soma_Detection(V3DPluginCallback2& cb);
    ~Soma_Detection()
    {
        delete[] imageSize;
    }
    bool saveConfiguration(const QString &savePath);
    void WHOLE_BRAIM_SOMA_DETECTION();
    bool Candidate_block_screening(const QString &image_path, int blockSize);
    bool ObtainALLimagePATH(const QString &image_path);

    bool Soma_location(const QString &image_path, int blockSize);
    //bool readStartMarkers(const QString& markerFile);
    //bool ObtainALLBlockPosition(int blockSize);
};

char* GetCWD();
bool whole_brain_candidate_block_screening(V3DPluginCallback2 &callback, V3DPluginArgList &args);
//string whole_brain_candidate_block_screening(V3DPluginCallback2 &callback, V3DPluginArgList &args, QProcess &process);
bool whole_brain_soma_location(V3DPluginCallback2 &callback, V3DPluginArgList &args);

#endif //_SOMA_DETECTION_