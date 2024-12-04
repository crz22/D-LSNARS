#include "io2.h"

bool getLandmarkTeraFly1(V3DPluginCallback2& callback,V3DPluginArgList &input, V3DPluginArgList &output){
    vector<char *> infiles, outfiles;
    if (input.size() >= 1) infiles = *((vector<char *> *) input.at(0).p);
    if (output.size() >= 1) outfiles = *((vector<char *> *) output.at(0).p);

    if (infiles.empty() || outfiles.empty()) return false;
    vector<MyMarker *> markers;
    for (const auto &m: callback.getLandmarkTeraFly()) {
        markers.push_back(new MyMarker(m.x, m.y, m.z));
    }
    if (markers.empty()) return false;
    list<string> nullinfostr;
    return saveMarker_file1(outfiles[0], markers);
}


bool saveMarker_file1(const char* marker_file, vector<MyMarker *> &outmarkers) {
    cout << "save " << outmarkers.size() << " markers to file " << marker_file << endl;
    ofstream ofs(marker_file, ios::app);

    if (ofs.fail()) {
        cout << "open marker file error" << endl;
        return false;
    }

//    ofs << "#x,y,z,radius,shape,name,comment,color_r,color_g,color_b" << endl;
    for (auto &outmarker: outmarkers) {
        ofs << outmarker->x << "," << outmarker->y << "," << outmarker->z << ","
            << 0 << "," << 0 << "," << "" << "," << "" << "," << 255 << "," << 0 << "," << 0 << endl;
    }
    ofs.close();
    return true;
}

bool saveSWCFile(const string savefile, NodeList& neuronTree, bool verbose)
{
    QString filePath = QString::fromStdString(savefile);
    QFile file(filePath);
    if (file.exists()) file.remove();
    ofstream ofs(filePath.toStdString(), ios::binary);
    if (ofs.fail()) {
        if (verbose) std::cout<<"failed to save " << filePath.toStdString() <<std::endl;
        return false;
    }
    V3DLONG nums = 0;
    unordered_map<NeuronNode*, V3DLONG> ind;
    for (NeuronNode* node: neuronTree) ind.insert(pair<NeuronNode*, V3DLONG>(node, ++nums));
    ofs<<"# name "<<qPrintable(filePath)<<std::endl;
    ofs<<"# comment "<<std::endl;
    ofs<<"# n, type, x, y, z, radius, parent"<<std::endl;
    nums = 0;
    for (NeuronNode* node: neuronTree)
    {
        V3DLONG parent_id = (node->parent && ind.count(node->parent))? ind[node->parent]: -1;
        ofs<<++nums<<" "<<node->type<<" "<<node->x<<" "<<node->y<<" "<<node->z<<" "<<node->radius<<" "<<parent_id<<std::endl;
    }
    if (verbose) std::cout<<"[Info] Marker num = "<<nums<<", save swc file to "<<qPrintable(filePath)<<std::endl;
    ofs.close();
    return true;
}

V3DLONG* getDimTeraFly1(V3DPluginCallback2& callback,QString &input){
    V3DLONG* imageSize = new V3DLONG[4]();
    callback.getDimTeraFly(input.toStdString(), imageSize);  
    cout <<"image size: "<< imageSize[0] << " " << imageSize[1] << " " << imageSize[2] << " " << imageSize[3] <<endl;
    //delete[] imageSize;
    return imageSize;
}

bool getSubVolumeFromTeraFly1(V3DPluginCallback2 &callback, char *imagePath, Image4DSimple &subVolumeImage, V3DLONG xb,
                             V3DLONG xe, V3DLONG yb, V3DLONG ye, V3DLONG zb, V3DLONG ze,V3DLONG *originSize) {

    V3DLONG subVolumeXdim, subVolumeYdim, subVolumeZdim, subVolumeCdim;
    subVolumeXdim = (xe < originSize[0]) ? xe - xb : originSize[0] - xb;
    subVolumeYdim = (ye < originSize[1]) ? ye - yb : originSize[1] - yb;
    subVolumeZdim = (ze < originSize[2]) ? ze - zb : originSize[2] - zb;
    subVolumeCdim = originSize[3];

    unsigned char *total4DData = callback.getSubVolumeTeraFly(imagePath, xb, xb + subVolumeXdim, yb,
                                                               yb + subVolumeYdim, zb, zb + subVolumeZdim);
    subVolumeImage.setData(total4DData, subVolumeXdim, subVolumeYdim, subVolumeZdim, subVolumeCdim,
                           V3D_UINT8);
    subVolumeImage.setOriginX((double) xb);
    subVolumeImage.setOriginY((double) yb);
    subVolumeImage.setOriginZ((double) zb);
    //delete[] originSize;
    return true;
}

bool normalization(Image4DSimple *image) {
    V3DLONG imageSize = image->getTotalUnitNumber();
    auto pRawData = image->getRawData();
    unsigned char maxValue = 0, minValue = 255;
    for (V3DLONG i = 0; i < imageSize; ++i) {
        maxValue = std::max(pRawData[i], maxValue);
        minValue = std::min(pRawData[i], minValue);
    }
    if (maxValue == minValue) {
        v3d_msg("max value = min value, unable to normalize", false);
        return false;
    }
    for (V3DLONG i = 0; i < imageSize; ++i) {
        pRawData[i] = int(pRawData[i] - minValue) * 255 / (maxValue - minValue);
    }
    return true;
}

bool readSWCtoNodeList(const string filePath, NodeList &nt){
    //
    ifstream ifs(filePath);
    if (ifs.fail()) {
        cout << "[Info] Open " << filePath << " failed." << endl;
        return false;
    }
    //Read the reconstructed block .swc file
    nt.clear();
    unordered_map<int, NeuronNode *> marker_map;
    unordered_map<NeuronNode *, int> parid_map;
    for (; ifs.good(); ifs.ignore(1024, '\n')) 
    {
        if (ifs.peek() == '#' || ifs.eof()) continue;
        int id = -1;
        int par_id = -1;
        ifs >> id;
        NeuronNode *pNode = new NeuronNode;
        marker_map[id] = pNode;
        ifs >> pNode->type;
        ifs >> pNode->x;
        ifs >> pNode->y;
        ifs >> pNode->z;
        ifs >> pNode->radius;
        ifs >> par_id;
        parid_map[pNode] = par_id;
        nt.addNode(pNode);
    }
    ifs.close();

    for (NeuronNode *pNode: nt) 
    {
        int parid = parid_map[pNode];
        if (parid == -1) continue;
        pNode->parent = marker_map[parid];
        pNode->parent->children.push_back(pNode);
    }
    return true;
}

void file_copy(string src, string target){
    QString sourceFile = QString::fromStdString(src);
    QString targetFile = QString::fromStdString(target);
    QFile file(sourceFile);

    if (file.copy(targetFile))  qDebug() << "File copied successfully."<<endl;
    else qDebug() << "Failed to copy file:" << file.errorString()<<endl;
    return ;
}