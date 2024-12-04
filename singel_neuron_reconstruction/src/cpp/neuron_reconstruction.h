#ifndef _NEURON_RECONSTRUCTION_
#define _NEURON_RECONSTRUCTION_

#include <iostream>
#include <fstream>
#include <string>
#include <direct.h>
#include <QtGui>
#include <QtCore/QVariant>
#include <QDateTime>
#include <vector>
#include "v3d_interface.h"
#include "basic_surf_objs.h"
#include "io2.h"
using namespace std;
enum Direction{LeftSide = 0, RightSide, UpSide, DownSide, OutSide, InSide};

class Neuron_Reconstruction: public QObject
{
    Q_OBJECT

public:
    QString imagePath;
    QString markerPath;
    QString outputFolderPath;
    QString pytorchPath;

    QString traceMethod;
    QString segmentMethod;
    int blockSize;
    float margin_lamd;
    long long MaxpointNUM;
    int marginSize;
    int node_step;
    int branch_MAXL;
    float Angle_T,Lamd;
    int min_branch_length;

    bool terafly = true;

    BlockSimpleList blockList, allTargetList;

private:
    V3DPluginCallback2& callback;
    QString tempFolderPath;
    QString configurationPath;
    QString finalSWCfile;
    string codePath;

    vector<string> parasString;
    V3DLONG* imageSize = new V3DLONG[4]();
    int somatype = 2;  //points connect with soma
       unordered_map<string, BlockSimple*> blockMap;

public:
    explicit Neuron_Reconstruction(V3DPluginCallback2& cb);
    ~Neuron_Reconstruction()
    {
        delete[] imageSize;
    }
    bool saveConfiguration(const QString &savePath);
    void SINGEL_NEURON_RCONSTRUCT();
    bool readStartMarkers(const QString &markerFile);
    void Block_boundary_adjust(BlockSimple *pBlock);
    bool Block_Neuron_Reconstruct(BlockSimple *currentTarget,V3DPluginArgList &args, NodeList &blockNeuronTree,
                                        string coordinateString,V3DLONG OriginX, V3DLONG OriginY, V3DLONG OriginZ);
    bool readBlockSWC(const QString &coordinate, const QString &postfix, NodeList &nt, bool verbose = FALSE);
    bool Reconstruction_algorithm(V3DPluginArgList &args, bool isStartBlock);
    void SearchNodesOnBoundary(BlockSimple *Target, NodeList &nt, int marginSize);
    void findTips(NodeList &nt, V3DLONG start, V3DLONG end, Direction direction);
    bool pruneTinyBranch(int length, NodeList *nodeList, BlockSimple *blockSimple);
    void SearchNearBlock(BlockSimple *centralblock, BlockSimpleList &candidateNeighbours);
    bool connect(BlockSimpleList &candidateGroups, NodeList &neuronTree, NodeList &connectedSegs, double thresDist = 5.0); 
    bool extractNeuronSegment(NodeList &nt, NeuronNode *pNode, NodeList &neuronSeg);
    bool setNewRoot(NeuronNode *newRoot);
    bool findNodeInBranch(NeuronNode *branchNode,NeuronNode *searchNode);
    void search26Neighbours(BlockSimple *currentTarget, BlockSimpleList &newTargetList, NodeList &tipList); 
    bool saveFinalNeuronTree(const QString &savePath);
    bool adjustFinalNeuronTree(const QString &swcPath);
    bool pruneFinalBranch(int length, NodeList *NTree);
};

char* GetCWD();
bool block_segmentation(V3DPluginCallback2 &callback, V3DPluginArgList &args,string input_image, const QString method);
bool SPE_DNR(V3DPluginCallback2 &callback, V3DPluginArgList &args, bool isStartBlock);
bool resample(V3DPluginCallback2 &callback, const string &swcfile, string stepLen, const string &resampledSwcFile);
NeuronNode *interpolateNodeOnBoundary(NeuronNode *src, double boundary, Direction direction);
bool error_reconstruct(V3DPluginArgList &args, string input_path);
//bool resample(const string &swcfile, string stepLen, const string &resampledSwcFile);

#endif //_NEURON_RECONSTRUCTION_

