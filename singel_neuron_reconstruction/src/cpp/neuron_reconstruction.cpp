#include "neuron_reconstruction.h"

Neuron_Reconstruction :: Neuron_Reconstruction(V3DPluginCallback2& cb):
callback(cb)
{   
    cout<<"Neuron_Reconstruction start"<<endl;
}

void Neuron_Reconstruction :: SINGEL_NEURON_RCONSTRUCT(){
    // obtain code current path
    codePath = GetCWD();
    // bulid result save files
    if (outputFolderPath.isEmpty()) outputFolderPath = "temp";
    QDateTime dateTime = QDateTime::currentDateTime();
    outputFolderPath.append("/" + dateTime.toString("yyyy-MM-dd/hh_mm_ss"));
    QDir outputDir(outputFolderPath);
    if (!outputDir.exists()) outputDir.mkpath(".");
    
    // bulid temp save files
    QDir imageDir(imagePath);
    imageDir.cdUp();
    outputDir.mkpath(imageDir.dirName() + "_tmp");
    tempFolderPath = outputFolderPath + "/" + imageDir.dirName() + "_tmp";

    finalSWCfile = QString("%1/%2_%3").arg(outputFolderPath, imageDir.dirName(), traceMethod);
    finalSWCfile.append(".swc");
    
    //path to save method parameteras 
    configurationPath = outputFolderPath + "/configuration.yaml";
    saveConfiguration(configurationPath);

    //set parameters and extract image blocks for neuron reconstruct
    V3DPluginArgList tracingArgsList;
    tracingArgsList.push_back(V3DPluginArgItem());
    tracingArgsList.push_back(V3DPluginArgItem());
    vector<char *> subImage(1, nullptr), paras;
    tracingArgsList.front().p = &subImage;
    tracingArgsList.back().p = &paras;    //tracingArgsList = [subImage,paras]

    parasString.emplace_back(pytorchPath.toStdString());
    parasString.emplace_back(configurationPath.toStdString());
    parasString.emplace_back(tempFolderPath.toStdString());
    for (auto &para: parasString) {
        paras.push_back(const_cast<char *>(para.data()));
    }
    
    cout<<"marginlamd: "<<margin_lamd<<endl;
    marginSize = margin_lamd * blockSize;
    cout<<"marginSize: "<<marginSize<<endl;
    //************************************************************************************************/
    //Calculate program runtime
    QElapsedTimer timer{};
    timer.start(); 

    if (terafly){
        //read start marker
        if (markerPath.isEmpty()) 
        {
            string markerfilePath = (outputFolderPath + "/start.marker").toStdString();
            V3DPluginArgList inputArgsList, outputArgsList;
            vector<char *> unused{nullptr};
            vector<char *> output{const_cast<char *>(markerfilePath.data())};
            inputArgsList.push_back(V3DPluginArgItem());
            inputArgsList.front().p = &unused;
            outputArgsList.push_back(V3DPluginArgItem());
            outputArgsList.front().p = &output;
            
            getLandmarkTeraFly1(callback,inputArgsList,outputArgsList);
            if (!readStartMarkers(markerfilePath.data()) || allTargetList.empty())
            {
                cout<<"Error: no find marker in whole brain image"<<endl;
                return ;
            } 
            //Leave only one starting point
            while (allTargetList.size() > 1) 
            {
                delete allTargetList.front();
                allTargetList.pop_front();
            }
        }
        else 
        {
            if (!readStartMarkers(markerPath)) 
            {
                cout << "ERROR: failed to read .marker: " << markerPath.toStdString() << endl;
                return ;
            }
            if (allTargetList.empty()) 
            {
                cout << "ERROR: Empty .marker file " << markerPath.toStdString() << endl;
                return ;
            }
        }
        //get size of big image
        imageSize = getDimTeraFly1(callback,imagePath);
    }
    else{cout<<"No terafly !!!"<<endl;}

    //set start block 
    for (BlockSimple *pStartBlock: allTargetList) 
    {
        if (pStartBlock == allTargetList.front()) 
        {
            pStartBlock->originX -= blockSize / 2;
            pStartBlock->originY -= blockSize / 2;
            pStartBlock->originZ -= blockSize / 2;
        }
        else 
        {  
           cout<<"ERROR: start point more than one"<<endl;
           return ;
        }
        pStartBlock->isStartBlock = true;
    }

    //Start tracking the neural structure in whole brain image
    V3DLONG finalNeuronTreeSize = 0;
    
    //Start tracking the neural structure in each image block
    V3DLONG traceCount = blockList.size();
    for (; !allTargetList.empty() && finalNeuronTreeSize < MaxpointNUM; allTargetList.pop_front()) 
    {
        //trace the neural structure in current image block
        cout << "[Info] Remaining target(s): " << allTargetList.size() << endl;
        BlockSimple *currentTarget = allTargetList.front();
        Block_boundary_adjust(currentTarget);
        V3DLONG OriginX, OriginY, OriginZ;
        OriginX = currentTarget->originX;
        OriginY = currentTarget->originY;
        OriginZ = currentTarget->originZ;
        string coordinateString = QString("x%1_y%2_z%3").arg(OriginX).arg(OriginY).arg(OriginZ).toStdString();
        cout << "[Info] CurrentTarget: " << coordinateString << endl;
        
        //Determine whether the image block has been reconstructed,
        //If not, extract the image block for reconstruction; otherwise read swc file to blockNeuronTree
        NodeList blockNeuronTree;
        if (!readBlockSWC(QString(coordinateString.c_str()), QString(".tif_not_connect.swc"), blockNeuronTree)){
            if(!Block_Neuron_Reconstruct(currentTarget, tracingArgsList, blockNeuronTree, coordinateString, OriginX, OriginY, OriginZ))
            {
                cout<<"ERROE: Neuron block reconstruction fail"<<endl;
                return ;
            }
        }
        
        if (blockNeuronTree.isEmpty()) 
        {
            delete currentTarget;
            currentTarget = nullptr;
            std::cout << "[Info] Empty target.\n" << std::endl;
            continue;
        }
        //Connect the current SWC with the overall result
        NodeList connectedNeuronSegs;
        //cout<<"blockNeuronTree size1: "<<blockNeuronTree.size()<<endl;
        if (currentTarget->isStartBlock) {
            for (NeuronNode *node: blockNeuronTree) {
                if(node->type == somatype)
                  connectedNeuronSegs.addNode(node);   
            }    
            currentTarget->isStartBlock = false;
            //for (NeuronNode *node: connectedNeuronSegs) 
            //    blockNeuronTree.removeOne(node);
            //connectedNeuronSegs.extractAllNodes(blockNeuronTree);
            //cout<<"blockNeuronTree size2: "<<blockNeuronTree.size()<<endl;
        } 
        else 
        {
            cout<<"connection"<<endl;
            // 26 neighbours connection
            BlockSimpleList candidateNeighbours;
            SearchNearBlock(currentTarget,candidateNeighbours);
            //connect(candidateNeighbours, blockNeuronTree);
            if (!connect(candidateNeighbours, blockNeuronTree, connectedNeuronSegs))
                cout << "connect failed." << endl;   
            
            //if (coordinateString == "x14670_y26235_z3904")
            //    return;
        }
        
        if (connectedNeuronSegs.isEmpty()) 
        {
            delete currentTarget;
            currentTarget = nullptr;
            string waitconnectSwcFile = (tempFolderPath + "/").toStdString() + coordinateString + ".tif_not_connect.swc";
            saveSWCFile(waitconnectSwcFile, blockNeuronTree);
            cout << "[Info] Non-connected target.\n" << endl;
            continue;
        }
        
        //Find the Terminalpoints of the neuron
        NodeList newTerminalPoints;
        for (NeuronNode *pNode: connectedNeuronSegs) 
        {
            if (!pNode->children.empty()) continue; 
            newTerminalPoints.push_back(pNode);
        }

        // 26 neighbours search
        search26Neighbours(currentTarget, allTargetList, newTerminalPoints);
        // record reconstructed block
        NodeList *pCurrentTargetNodeList = nullptr;
        if (blockMap.find(coordinateString) == blockMap.end()) 
        {
            pCurrentTargetNodeList = new NodeList();
            currentTarget->pBlockNodeList = pCurrentTargetNodeList;
            blockMap[coordinateString] = currentTarget;
            blockList.push_back(currentTarget);
        } 
        else 
        {
            pCurrentTargetNodeList = blockMap[coordinateString]->pBlockNodeList;
            delete currentTarget;
            currentTarget = nullptr;
        }
        //
        finalNeuronTreeSize += connectedNeuronSegs.size();
        while (!connectedNeuronSegs.empty()) 
        {
            pCurrentTargetNodeList->addNode(connectedNeuronSegs.front());
        }

        //string waitconnectSwcFile = coordinateString + ".tif_not_connect.swc";
        string waitconnectSwcFile = (tempFolderPath + "/").toStdString() + coordinateString + ".tif_not_connect.swc";
        saveSWCFile(waitconnectSwcFile, blockNeuronTree);

        while (!blockNeuronTree.empty()) 
        {
            delete blockNeuronTree.front();
        }
        if (traceCount % 10 == 0){
            saveFinalNeuronTree(finalSWCfile);
            //NodeList FianlNeuronTree;
            //string stepLen = "5";  //5
            //bool resampleDone = resample(callback, finalSWCfile.toStdString(), stepLen, finalSWCfile.toStdString());
            //if (!resampleDone || !readSWCtoNodeList(finalSWCfile.toStdString(), FianlNeuronTree)){
            //    cout<<"ERROR: FianlNeuronTree resampled fail."<<endl;
            //    return ;
            //}
            //finalNeuronTreeSize = FianlNeuronTree.size();
        }
        cout << "[Info] Traced " << ++traceCount << " Count(s) successfully." << endl;
        cout << "[Info] Current neuron tree size: " << finalNeuronTreeSize << endl;
    }
    
    saveFinalNeuronTree(finalSWCfile);
    adjustFinalNeuronTree(finalSWCfile);
    
    qint64 endTime = timer.nsecsElapsed();
    float endTime_h = endTime/1e9/3600;
    cout<<"Neuron reconstruction finish"<<endl;
    cout<<"cost time: "<<endTime_h<<" h"<<endl;
    return ;
}

bool Neuron_Reconstruction :: Block_Neuron_Reconstruct(BlockSimple *currentTarget,V3DPluginArgList &args, NodeList &blockNeuronTree, 
                                    string coordinateString,V3DLONG OriginX, V3DLONG OriginY, V3DLONG OriginZ ){
    cout << "[Info] Tracing..." << endl;
    string imagepath_std = imagePath.toStdString();
    //extract and save image block from whole brain image
    //Image blocks are named with the coordinates of the upper left point
    string subImagePath = QString("%1/x%2_y%3_z%4.tif").arg(tempFolderPath).arg(OriginX).arg(OriginY).arg(OriginZ).toStdString();
    Image4DSimple subImage = Image4DSimple();
    if (terafly) 
    {
        if (!getSubVolumeFromTeraFly1(callback, const_cast<char *>(imagepath_std.data()), subImage,
                                        OriginX, OriginX + currentTarget->blockSizeX,
                                        OriginY, OriginY + currentTarget->blockSizeY,
                                        OriginZ, OriginZ + currentTarget->blockSizeZ,imageSize)) 
        {
            cerr << "[Error] Failed to get subvolume from TeraFly." << std::endl;
            return false;
        }
    } 
    else {cout<<"ERROR: NOT terafly image"<<endl;}

    //normalization(&subImage);

    callback.saveImage(&subImage, const_cast<char *>(subImagePath.data()));
    //image block segments
    if(!block_segmentation(callback,args,subImagePath,segmentMethod))
    {
        cout<<"ERROR: block segment fail"<<endl;
    };
    
    (static_cast<vector<char *> *>(args.front().p))->front() = const_cast<char *>(subImagePath.data());
    
    // Call reconstruction algorithm
    ; 
    if (!Reconstruction_algorithm(args,currentTarget->isStartBlock)) {
        cout<<"ERROR: Block neuron reconstruct fail"<<endl;
        return false;
    }
    
    // resample and read swc file to blockNeuronTree
    string rawSwcFile = subImagePath + ".swc";
    string resampledSwcFile = subImagePath + "_resampled.swc", stepLen = "5";
    bool resampleDone = resample(callback, rawSwcFile, stepLen, resampledSwcFile);
    if (!resampleDone || !readBlockSWC(QString(coordinateString.c_str()), QString(".tif_resampled.swc"),blockNeuronTree))
        return false;
    
    // prune(blockNeuronTree): Delete branches with fewer than 3 nodes
    string  removeTipsSwcFile = resampledSwcFile + "_removeTips.swc";
    SearchNodesOnBoundary(currentTarget,blockNeuronTree,marginSize);
    while (pruneTinyBranch(min_branch_length, &blockNeuronTree, currentTarget));
    saveSWCFile(removeTipsSwcFile, blockNeuronTree);

    //Convert local coordinates to global coordinates
    for (NeuronNode *m: blockNeuronTree) 
    {
        m->x += OriginX;
        m->y += OriginY;
        m->z += OriginZ;
    }
    string waitconnectSwcFile = subImagePath + "_not_connect.swc";
    saveSWCFile(waitconnectSwcFile,blockNeuronTree);
    //file_copy(removeTipsSwcFile,waitconnectSwcFile);
    if (blockNeuronTree.size()>=10000)
        return true;
    if (!error_reconstruct(args, waitconnectSwcFile) || !readBlockSWC(QString(coordinateString.c_str()), QString(".tif_not_connect.swc"),blockNeuronTree))
        return false;
    return true;
}

bool Neuron_Reconstruction :: readBlockSWC(const QString &coordinate, const QString &postfix, NodeList &nt, bool verbose){
    string filePath = (tempFolderPath + "/" + coordinate + postfix).toStdString();
    //
    ifstream ifs(filePath);
    if (ifs.fail()) {
        if (verbose) 
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

void Neuron_Reconstruction :: Block_boundary_adjust(BlockSimple *pBlock){
    pBlock->originX = (pBlock->originX > 0) ? pBlock->originX : 0;
    pBlock->originY = (pBlock->originY > 0) ? pBlock->originY : 0;
    pBlock->originZ = (pBlock->originZ > 0) ? pBlock->originZ : 0;
    pBlock->blockSizeX = (pBlock->originX + blockSize) < imageSize[0] ? blockSize : imageSize[0] - pBlock->originX;
    pBlock->blockSizeY = (pBlock->originY + blockSize) < imageSize[1] ? blockSize : imageSize[1] - pBlock->originY;
    pBlock->blockSizeZ = (pBlock->originZ + blockSize) < imageSize[2] ? blockSize : imageSize[2] - pBlock->originZ;
}

bool Neuron_Reconstruction :: saveConfiguration(const QString &savePath) {
    QFile outFile(savePath);
    if (outFile.exists()) outFile.remove();
    if (!outFile.open(QFile::ReadWrite)) return false;
    QTextStream textStream(&outFile);
    textStream << "traceMethod: "<< traceMethod<<endl;
    textStream << "segmentMethod: "<< segmentMethod<<endl;
    textStream << "blocksize: "<< blockSize<<endl;
    textStream << "node_step: "<< node_step<<endl; 
    textStream << "branch_MAXL: "<< branch_MAXL<<endl;
    textStream << "Angle_T: " << Angle_T << endl;
    textStream << "Lamd: " << Lamd<< endl;
    //textStream << "OutputPath: " << QFileInfo(savePath).absolutePath() << endl;
    //textStream << "OutputPath: " << QFileInfo(savePath).absolutePath() << endl;
    outFile.close();
    return true;
}

bool Neuron_Reconstruction :: readStartMarkers(const QString &markerFile) {
    ifstream ifs(markerFile.toStdString(), ios::binary);
    if (ifs.fail()) return false;
    for (; ifs.good(); ifs.ignore(1024, '\n')) 
    {
        if (ifs.peek() == '#' || ifs.eof()) continue;
        BlockSimple *pStartMarker = new BlockSimple;
        ifs >> pStartMarker->originX;
        ifs.ignore(10, ',');
        ifs >> pStartMarker->originY;
        ifs.ignore(10, ',');
        ifs >> pStartMarker->originZ;
        ifs.ignore(10, ',');
        allTargetList.push_back(pStartMarker);
    }
    ifs.close();
    return true;
}

bool Neuron_Reconstruction :: Reconstruction_algorithm(V3DPluginArgList &args, bool isStartBlock){
    string METHOD = traceMethod.toStdString();
    if (METHOD == "SPE_DNR"){
        if(!SPE_DNR(callback,args,isStartBlock)){
            cout<<"Error: SPE_DNR fail"<<endl;
            return false;
        }
        return true;
    }
    else return false;

}

void Neuron_Reconstruction :: SearchNodesOnBoundary(BlockSimple *Target, NodeList &nt, int marginSize){
    if (Target->originX != 0)
        findTips(nt, 0, marginSize, Direction::LeftSide);

    if (Target->originX + Target->blockSizeX != imageSize[0]) 
        findTips(nt,Target->blockSizeX - marginSize, Target->blockSizeX,Direction::RightSide);
       
    if (Target->originY != 0)
        findTips(nt,0, marginSize, Direction::UpSide);

    if (Target->originY + Target->blockSizeY != imageSize[1])
        findTips(nt, Target->blockSizeY - marginSize, Target->blockSizeY, Direction::DownSide);

    if (Target->originZ != 0) 
        findTips(nt, 0, marginSize, Direction::OutSide);

    if (Target->originZ + Target->blockSizeZ != imageSize[2])
        findTips(nt, Target->blockSizeZ - marginSize, Target->blockSizeZ, Direction::InSide);
}

void Neuron_Reconstruction :: findTips(NodeList &nt, V3DLONG start, V3DLONG end, Direction direction) {
    double boundary = 0;
    NodeList tipList;
    //find the points in the boundary area (overlapping area of two image blocks)
    switch (direction) {
        case Direction::LeftSide:
            for (NeuronNode *m: nt) {
                if (m->x >= end) continue;
                tipList.push_back(m);
            }
            boundary = end;
            break;
        case Direction::RightSide:
            for (NeuronNode *m: nt) {
                if (m->x <= start) continue;
                tipList.push_back(m);
            }
            boundary = start;
            break;
        case Direction::UpSide:
            for (NeuronNode *m: nt) {
                if (m->y >= end) continue;
                tipList.push_back(m);
            }
            boundary = end;
            break;
        case Direction::DownSide:
            for (NeuronNode *m: nt) {
                if (m->y <= start) continue;
                tipList.push_back(m);
            }
            boundary = start;
            break;
        case Direction::OutSide:
            for (NeuronNode *m: nt) {
                if (m->z >= end) continue;
                tipList.push_back(m);
            }
            boundary = end;
            break;
        case Direction::InSide:
            for (NeuronNode *m: nt) {
                if (m->z <= start) continue;
                tipList.push_back(m);
            }
            boundary = start;
            break;
    }
    //Insert point in the boundary
    NodeList nonTipChildrenList;
    for (NeuronNode *tip: tipList) 
    {   //Determine whether the child nodes of overlapping area nodes are in the overlapping area
        for (NeuronNode *child: tip->children) 
        {
            if (tipList.indexOf(child) != -1) continue;
        //if the child node is not in the overlapping area, insert a point on the boundary between the parent and child nodes
            nt.addNode(interpolateNodeOnBoundary(child, boundary, direction));
        }
        //Determine whether the parent nodes of overlapping area nodes are in the overlapping area
        if (tip->parent == nullptr) continue;
        if (tipList.indexOf(tip->parent) != -1) continue;
        //if the parent node is not in the overlapping area, insert a point on the boundary between the parent and child nodes
        nt.addNode(interpolateNodeOnBoundary(tip, boundary, direction));
    }

    //clear tipList
    for (; !tipList.empty(); tipList.pop_front()) {
        NeuronNode *node = tipList.front();
        delete node;
    }
}

NeuronNode *interpolateNodeOnBoundary(NeuronNode *src, double boundary, Direction direction) {
    NeuronNode *oldParent = src->parent;
    NeuronNode *interpolateNode = new NeuronNode();
    interpolateNode->x = oldParent->x - src->x;
    interpolateNode->y = oldParent->y - src->y;
    interpolateNode->z = oldParent->z - src->z;

    double scale;
    if (direction == Direction::LeftSide || direction == Direction:: RightSide)
        scale = (boundary - src->x) / interpolateNode->x;
    else if (direction == Direction::UpSide || direction == Direction::DownSide)
        scale = (boundary - src->y) / interpolateNode->y;
    else if (direction == Direction::OutSide || direction == Direction::InSide)
        scale = (boundary - src->z) / interpolateNode->z;

    interpolateNode->x *= scale;
    interpolateNode->y *= scale;
    interpolateNode->z *= scale;

    interpolateNode->x += src->x;
    interpolateNode->y += src->y;
    interpolateNode->z += src->z;

    interpolateNode->radius = oldParent->radius / 2 + src->radius / 2;
    interpolateNode->type = oldParent->type;
    oldParent->children.removeOne(src);
    oldParent->children.push_back(interpolateNode);
    interpolateNode->parent = oldParent;
    src->parent = interpolateNode;
    interpolateNode->children.push_back(src);
    return interpolateNode;
}

bool Neuron_Reconstruction :: pruneTinyBranch(int length, NodeList *nodeList, BlockSimple *blockSimple) {
    NodeList tinyBranchBack, tinyBranchFront;
    if (nodeList != nullptr) 
    {
        for (auto node: *(nodeList)) 
        {
            if (!node->children.empty()) continue;
            if (abs(node->x - marginSize) < 3 || abs(node->x - blockSimple->blockSizeX + marginSize) < 3) continue;
            if (abs(node->y - marginSize) < 3 || abs(node->y - blockSimple->blockSizeY + marginSize) < 3) continue;
            if (abs(node->z - marginSize) < 3 || abs(node->z - blockSimple->blockSizeZ + marginSize) < 3) continue;
            NeuronNode *branchPoint = node;
            int count = 0;
            while (branchPoint->parent && branchPoint->parent->children.size() < 2 && count < length) 
            {
                branchPoint = branchPoint->parent;
                count++;
            }
            if (count < length && branchPoint->radius<20) // && branchPoint->parent
            {
                tinyBranchBack.push_back(node);
                tinyBranchFront.push_back(branchPoint);
            }
        }
    } 
    else 
    {   /*
        for (auto block: this->blockList) 
        {
            for (auto node: *(block->pBlockNodeList)) 
            {
                if (!node->children.empty()) continue;
                NeuronNode *branchPoint = node;
                int count = 0;
                while (branchPoint->parent && branchPoint->parent->children.size() < 2 && count < length) {
                    branchPoint = branchPoint->parent;
                    count++;
                }
                if (count < length ) //&& branchPoint->parent
                {
                    tinyBranchBack.push_back(node);
                    tinyBranchFront.push_back(branchPoint);
                }
            }
        }
        */
       cout<<"pruneTinyBranch fail: nodelist is empty."<<endl;
       return false;
    }
    for (auto front: tinyBranchFront) 
    {
        if (front->parent == nullptr) continue;
        front->parent->children.removeOne(front);
        front->parent = nullptr;
    }
    for (auto back: tinyBranchBack) 
    {
        NeuronNode *next = nullptr;
        do {
            next = back->parent;
            delete back;
            back = next;
        } while (next != nullptr);
    }
    cout<< "[Info] pruned "<<tinyBranchFront.size()<<endl;
    if (tinyBranchFront.size()!=0) return true;
    else return false;
}

void Neuron_Reconstruction :: SearchNearBlock(BlockSimple *centralblock, BlockSimpleList &candidateNeighbours){
    V3DLONG OriginX, OriginY, OriginZ;
    OriginX = centralblock->originX;
    OriginY = centralblock->originY;
    OriginZ = centralblock->originZ;
    string neighbourCoordinate;

    for (V3DLONG zz = -1; zz <= 1; zz++) 
    {
        if (zz == -1 && OriginZ == 0) continue;
        if (zz == 1 && (OriginZ + centralblock->blockSizeZ == imageSize[2])) continue;
        V3DLONG zBlockSize = (zz == 1) ? centralblock->blockSizeZ : blockSize;

        for (V3DLONG yy = -1; yy <= 1; yy++) 
        {
            if (yy == -1 && OriginY == 0) continue;
            if (yy == 1 && (OriginY + centralblock->blockSizeY == imageSize[1])) continue;
            V3DLONG yBlockSize = (yy == 1) ? centralblock->blockSizeY : blockSize;

            for (V3DLONG xx = -1; xx <= 1; xx++) 
            {
                if (xx == -1 && OriginX == 0) continue;
                if (xx == 1 && (OriginX + centralblock->blockSizeX == imageSize[0])) continue;
                if (xx == 0 && yy == 0 && zz == 0) continue;
                V3DLONG xBlockSize = (xx == 1) ? centralblock->blockSizeX : blockSize;

                V3DLONG xCoordinate, yCoordinate, zCoordinate;
                xCoordinate = OriginX + xx * (xBlockSize - 2 * marginSize);
                xCoordinate = (xCoordinate < 0) ? 0 : xCoordinate;
                yCoordinate = OriginY + yy * (yBlockSize - 2 * marginSize);
                yCoordinate = (yCoordinate < 0) ? 0 : yCoordinate;
                zCoordinate = OriginZ + zz * (zBlockSize - 2 * marginSize);
                zCoordinate = (zCoordinate < 0) ? 0 : zCoordinate;
                neighbourCoordinate = QString("x%1_y%2_z%3").arg(xCoordinate).arg(yCoordinate)
                                        .arg(zCoordinate).toStdString();

                if (blockMap.find(neighbourCoordinate) != blockMap.end()){
                    candidateNeighbours.push_back(blockMap[neighbourCoordinate]);
                    //cout<<"neighbour: "<<neighbourCoordinate<<endl;
                }      
            }
        }
    }        
}

bool Neuron_Reconstruction :: connect(BlockSimpleList &candidateGroups, NodeList &neuronTree, NodeList &ConnectedSegs, double thresDist) {
    double fusionDist = 1.0;
    unordered_map<NeuronNode *,NeuronNode *> ConnectPoint;
    unordered_map<NeuronNode *,double> ConnectDist;
    //Find the nodes where the current reconstruction result can be connected to the reconstruction result of near blocks
    V3DLONG ntSize = 0;
    for (NeuronNode *pNode: neuronTree) {
            double minDist = 100;
            NeuronNode *pConnectNode = nullptr;
            //Find the nearest reconstruction point in nearby image blocks
            for (BlockSimple *candidate: candidateGroups) {
                for (NeuronNode *pCandidateNode: *(candidate->pBlockNodeList)) {
                    double dist = pNode->getDistanceTo(pCandidateNode);
                    if (dist > minDist) continue;
                    if (dist > thresDist) continue;
                    pConnectNode = pCandidateNode;
                    minDist = dist;  
                }
            }
            if (pConnectNode == nullptr) continue;
            ConnectPoint[pNode] = pConnectNode;
            ConnectDist[pNode] = minDist;
    }
    
    while(!ConnectPoint.empty())
    {
        NeuronNode *pNode;
        NeuronNode *pConnectNode;
        //Find the point with the smallest connection distance
        double minDist = 10e4;
        for (const auto& pair : ConnectDist)
        {
            if(pair.second<minDist){
                pNode = pair.first;
                minDist = pair.second;
            }
        }
        pConnectNode = ConnectPoint[pNode];
        ConnectPoint.erase(pNode);
        ConnectDist.erase(pNode);
        // extract the branch of pNode and
        NodeList connectedSeg;
        if (!extractNeuronSegment(neuronTree, pNode, connectedSeg)) return false;
        // changed parents point and children point
        if (pNode->parent != nullptr && !setNewRoot(pNode)) return false;
        //connect pNode with pConnectNode
        if (minDist <= fusionDist) 
        {   
            while (!pNode->children.empty()) 
            {
                NeuronNode *child = pNode->children.front();
                child->parent = pConnectNode;
                pConnectNode->children.push_back(child);
                pNode->children.pop_front();
            }
            delete pNode;
        }
        else 
        {  
            pNode->parent = pConnectNode;
            pConnectNode->children.push_back(pNode);
        }

        //Connect other points on this branch that may be connected to surrounding blocks
        NodeList CurBranchConnectPoint;
        for(NeuronNode *connectNode: connectedSeg)
        {
            if (ConnectPoint.find(connectNode) != ConnectPoint.end())
                CurBranchConnectPoint.push_back(connectNode);
        }
        while (!CurBranchConnectPoint.empty())
        {   
            minDist = 1e4;
            //Find the point with the smallest connection distance
            for(NeuronNode *candidateconnectNode: CurBranchConnectPoint)
            {
                if(ConnectDist[candidateconnectNode]<minDist)
                {   
                    minDist = ConnectDist[candidateconnectNode];
                    pNode = candidateconnectNode;
                }
            }
            pConnectNode = ConnectPoint[pNode];
            ConnectPoint.erase(pNode);
            ConnectDist.erase(pNode);
            CurBranchConnectPoint.removeOne(pNode);
            if (findNodeInBranch(pNode,pConnectNode)) continue;
            //changed parents point and children point
            if (pNode->parent != nullptr && !setNewRoot(pNode)) return false;

            if (minDist <= fusionDist) 
            {   
                while (!pNode->children.empty()) 
                {
                    NeuronNode *child = pNode->children.front();
                    child->parent = pConnectNode;
                    pConnectNode->children.push_back(child);
                    pNode->children.pop_front();
                }
                delete pNode;
            }
            else 
            {
                pNode->parent = pConnectNode;
                pConnectNode->children.push_back(pNode);
            }
        }
        /*
        int prunedNode = pruneBranch(min_branch_length, &connectedSeg);
        while (prunedNode)
        {
            cout << "[Info] Branch pruned " << prunedNode << " nodes." << endl;
            prunedNode = pruneBranch(min_branch_length, &connectedSeg);
        }
        */
        while (!connectedSeg.empty())
        {
            NeuronNode *node = connectedSeg.front();
            node->type = pConnectNode->type;
            ConnectedSegs.addNode(node);
        } 
    }
    return true;
}

bool Neuron_Reconstruction :: extractNeuronSegment(NodeList &nt, NeuronNode *pNode, NodeList &neuronSeg) {
    if (nt.indexOf(pNode) == -1) return false;
    while (pNode->parent) pNode = pNode->parent;
    neuronSeg.clear();
    NodeList nextList;
    nextList.push_back(pNode);
    while (!nextList.empty()) 
    {
        NeuronNode *node = nextList.front();
        neuronSeg.addNode(node);
        //nt.removeOne(node);
        for (NeuronNode *pChild: node->children) nextList.push_back(pChild);
        nextList.pop_front();
    } 
    return true;
}

bool Neuron_Reconstruction :: setNewRoot(NeuronNode *newRoot) {
    //exchanged parents point and children point
    if (newRoot == nullptr) return false;
    //if (newRoot->parent == nullptr) return true;
    NeuronNode *prev = nullptr, *cur = newRoot, *next = cur->parent;
    while (cur) 
    {   
        if (cur->parent) cur->parent->children.removeOne(cur);
        cur->parent = prev;
        if (cur->parent) cur->parent->children.push_back(cur);
        prev = cur;
        cur = next;
        if (next) next = next->parent;
    }
    return true;
}

bool Neuron_Reconstruction :: findNodeInBranch(NeuronNode *branchNode,NeuronNode *searchNode){
    while (branchNode->parent) branchNode = branchNode->parent;
    NodeList Branch;
    Branch.push_back(branchNode);
    while (!Branch.empty()) 
    {
        NeuronNode *node = Branch.front();
        if (node == searchNode) return TRUE;
        for (NeuronNode *pChild: node->children) Branch.push_back(pChild);
        Branch.pop_front();
    } 
    return FALSE;
}

void Neuron_Reconstruction :: search26Neighbours(BlockSimple *currentTarget, BlockSimpleList &newTargetList, NodeList &tipList) {
    vector<bool> searchFlag(27, false);
    for (NeuronNode *tip: tipList) 
    {
        V3DLONG xx = 0, yy = 0, zz = 0;
        if (abs(tip->x - currentTarget->originX - marginSize) < 3) xx = -1;
        else if (abs(tip->x - currentTarget->originX - currentTarget->blockSizeX + marginSize) < 3) xx = 1;
        if (abs(tip->y - currentTarget->originY - marginSize) < 3) yy = -1;
        else if (abs(tip->y - currentTarget->originY - currentTarget->blockSizeY + marginSize) < 3) yy = 1;
        if (abs(tip->z - currentTarget->originZ - marginSize) < 3) zz = -1;
        else if (abs(tip->z - currentTarget->originZ - currentTarget->blockSizeZ + marginSize) < 3) zz = 1;
        if (xx == 0 && yy == 0 && zz == 0) continue;

        if (xx) searchFlag[xx + 13] = true;
        if (yy) searchFlag[yy * 3 + 13] = true;
        if (zz) searchFlag[zz * 9 + 13] = true;
        if (xx && yy) searchFlag[xx + yy * 3 + 13] = true;
        if (xx && zz) searchFlag[xx + zz * 9 + 13] = true;
        if (yy && zz) searchFlag[yy * 3 + zz * 9 + 13] = true;
        if (xx && yy && zz) searchFlag[xx + yy * 3 + zz * 9 + 13] = true;
    }

    for (V3DLONG zz = -1; zz <= 1; zz++) 
    {
        if (zz == -1 && currentTarget->originZ == 0) continue;
        if (zz == 1 && ((V3DLONG) (currentTarget->originZ + currentTarget->blockSizeZ) == imageSize[2])) continue;
        V3DLONG zBlockSize = (zz == 1) ? currentTarget->blockSizeZ : blockSize;
        
        for (V3DLONG yy = -1; yy <= 1; yy++) 
        {
            if (yy == -1 && currentTarget->originY == 0) continue;
            if (yy == 1 && ((V3DLONG) (currentTarget->originY + currentTarget->blockSizeY) == imageSize[1])) continue;
            V3DLONG yBlockSize = (yy == 1) ? currentTarget->blockSizeY : blockSize;
           
            for (V3DLONG xx = -1; xx <= 1; xx++) 
            {
                if (xx == -1 && currentTarget->originX == 0) continue;
                if (xx == 1 && ((V3DLONG) (currentTarget->originX + currentTarget->blockSizeX) == imageSize[0])) continue;
                if (!searchFlag[xx + yy * 3 + zz * 9 + 13]) continue;
                
                V3DLONG xBlockSize = (xx == 1) ? currentTarget->blockSizeX : blockSize;
                BlockSimple *newTarget = new BlockSimple();
                newTarget->originX = currentTarget->originX + xx * (xBlockSize - 2 * marginSize);
                newTarget->originY = currentTarget->originY + yy * (yBlockSize - 2 * marginSize);
                newTarget->originZ = currentTarget->originZ + zz * (zBlockSize - 2 * marginSize);
                
                if (xx == 0 && (blockSize - currentTarget->blockSizeX > 0))
                    newTarget->blockSizeX = currentTarget->blockSizeX;
                else newTarget->blockSizeX = blockSize;
                if (yy == 0 && (blockSize - currentTarget->blockSizeY > 0))
                    newTarget->blockSizeY = currentTarget->blockSizeY;
                else newTarget->blockSizeY = blockSize;
                if (zz == 0 && (blockSize - currentTarget->blockSizeZ > 0))
                    newTarget->blockSizeZ = currentTarget->blockSizeZ;
                else newTarget->blockSizeZ = blockSize;

                if (newTarget->originX < 0) {
                    newTarget->originX = 0;
                    newTarget->blockSizeX = currentTarget->originX + 2 * marginSize;
                }
                if (newTarget->originY < 0) {
                    newTarget->originY = 0;
                    newTarget->blockSizeY = currentTarget->originY + 2 * marginSize;
                }
                if (newTarget->originZ < 0) {
                    newTarget->originZ = 0;
                    newTarget->blockSizeZ = currentTarget->originZ + 2 * marginSize;
                }

                if (newTarget->originX + newTarget->blockSizeX > imageSize[0])
                    newTarget->blockSizeX = imageSize[0] - newTarget->originX;
                if (newTarget->originY + newTarget->blockSizeY > imageSize[1])
                    newTarget->blockSizeY = imageSize[1] - newTarget->originY;
                if (newTarget->originZ + newTarget->blockSizeZ > imageSize[2])
                    newTarget->blockSizeZ = imageSize[2] - newTarget->originZ;
                if (newTarget->blockSizeX < 2 * marginSize || newTarget->blockSizeY < 2 * marginSize ||
                    newTarget->blockSizeZ < 2 * marginSize)
                    continue;
                newTargetList.push_back(newTarget);
             }
        }
    }
}

/*
bool Neuron_Reconstruction :: saveFinalNeuronTree(const QString &savePath){
    if (QFileInfo(savePath).exists()) QFile(savePath).remove();
    string filePath = savePath.toStdString();
    string logPath = outputFolderPath.toStdString() + "/log.txt";
    ofstream swcFile(filePath, ios::binary), logFile(logPath, ios::binary);
    if (swcFile.fail()) 
    {
        cout << "[ERROR]: open save file failed." << endl;
        return false;
    }
    if (logFile.fail()) 
    {
        cout << "[ERROR]: open log file failed." << endl;
        return false;
    }
    V3DLONG nums = 0;
    double minX = INFINITY, maxX = 0, minY = INFINITY, maxY = 0, minZ = INFINITY, maxZ = 0;
    unordered_map<NeuronNode *, V3DLONG> ind;
    for (BlockSimple *pBlock: blockList) 
    {
        for (NeuronNode *node: *(pBlock->pBlockNodeList)) 
        {
            if (minX > node->x) minX = node->x;
            if (maxX < node->x) maxX = node->x;
            if (minY > node->y) minY = node->y;
            if (maxY < node->y) maxY = node->y;
            if (minZ > node->z) minZ = node->z;
            if (maxZ < node->z) maxZ = node->z;
            ind.insert(pair<NeuronNode *, V3DLONG>(node, ++nums));

        }
    }
    swcFile << "# marker path " << qPrintable(markerPath) << endl;
    swcFile << "# name " << filePath << endl;
    swcFile << "# comment " << endl;
    swcFile << "# n, type, x, y, z, radius, parent" << endl;
    swcFile << "# x range: " << maxX - minX << ", y range: " << maxY - minY << ", z range: " << maxZ - minZ<< endl;

    logFile << "# image path" << endl;
    logFile << qPrintable(imagePath) << endl;
    logFile << "# n, type, x, y, z, radius, parent" << endl;
    logFile << "# @ blockX blockY blockZ blockSizeX blockSizeY blockSizeZ" << endl;

    nums = 0;
    for (BlockSimple *pBlock: blockList) 
    {
        logFile << "@ " << pBlock->originX << " " << pBlock->originY << " " << pBlock->originZ
                << " " << pBlock->blockSizeX << " " << pBlock->blockSizeY << " " << pBlock->blockSizeZ << std::endl;
        
        for (NeuronNode *node: *(pBlock->pBlockNodeList)) 
        {
            V3DLONG parent_id = (node->parent && ind.count(node->parent)) ? ind[node->parent] : -1;
            swcFile << ++nums << " " << node->type << " " << node->x << " " << node->y << " " << node->z << " "
                    << node->radius << " " << parent_id << endl;
            logFile << nums << " " << node->type << " " << node->x << " " << node->y << " " << node->z << " "
                    << node->radius << " " << parent_id << endl;
        }
    }
    cout << "[Info] marker num = " << nums << ", save swc file to " << filePath << endl;
    swcFile.close();
    logFile.close();
    return true;
}
*/

bool Neuron_Reconstruction :: saveFinalNeuronTree(const QString &savePath){
    if (QFileInfo(savePath).exists()) QFile(savePath).remove();
    string filePath = savePath.toStdString();
    string logPath = outputFolderPath.toStdString() + "/log.txt";
    ofstream swcFile(filePath, ios::binary), logFile(logPath, ios::binary);
    if (swcFile.fail()) 
    {
        cout << "[ERROR]: open save file failed." << endl;
        return false;
    }
    if (logFile.fail()) 
    {
        cout << "[ERROR]: open log file failed." << endl;
        return false;
    }
    V3DLONG nums = 0;
    double minX = INFINITY, maxX = 0, minY = INFINITY, maxY = 0, minZ = INFINITY, maxZ = 0;
    double rmax = 0;
    unordered_map<NeuronNode *, V3DLONG> ind;

    unordered_map<int, NeuronNode *> marker_map;
    unordered_map<NeuronNode *, int> parid_map;
    NeuronNode *soma_node;

    for (BlockSimple *pBlock: blockList) 
    {
        for (NeuronNode *node: *(pBlock->pBlockNodeList)) 
        {
            if (minX > node->x) minX = node->x;
            if (maxX < node->x) maxX = node->x;
            if (minY > node->y) minY = node->y;
            if (maxY < node->y) maxY = node->y;
            if (minZ > node->z) minZ = node->z;
            if (maxZ < node->z) maxZ = node->z;
            ind.insert(pair<NeuronNode *, V3DLONG>(node, ++nums));
            //find the soma point
            if (rmax < node->radius)
            {
                rmax = node->radius;
                soma_node = node;
            }
        }
    }
    //set the root point to soma node
    if (soma_node->parent != nullptr && !setNewRoot(soma_node)) return false;

    swcFile << "# marker path " << qPrintable(markerPath) << endl;
    swcFile << "# name " << filePath << endl;
    swcFile << "# comment " << endl;
    swcFile << "# n, type, x, y, z, radius, parent" << endl;
    swcFile << "# x range: " << maxX - minX << ", y range: " << maxY - minY << ", z range: " << maxZ - minZ<< endl;

    logFile << "# image path" << endl;
    logFile << qPrintable(imagePath) << endl;
    logFile << "# n, type, x, y, z, radius, parent" << endl;
    logFile << "# @ blockX blockY blockZ blockSizeX blockSizeY blockSizeZ" << endl;

    nums = 0;
    for (BlockSimple *pBlock: blockList) 
    {
        logFile << "@ " << pBlock->originX << " " << pBlock->originY << " " << pBlock->originZ
                << " " << pBlock->blockSizeX << " " << pBlock->blockSizeY << " " << pBlock->blockSizeZ << std::endl;
        
        for (NeuronNode *node: *(pBlock->pBlockNodeList)) 
        {
            V3DLONG parent_id = (node->parent && ind.count(node->parent)) ? ind[node->parent] : -1;
            swcFile << ++nums << " " << node->type << " " << node->x << " " << node->y << " " << node->z << " "
                    << node->radius << " " << parent_id << endl;
            logFile << nums << " " << node->type << " " << node->x << " " << node->y << " " << node->z << " "
                    << node->radius << " " << parent_id << endl;
        }
    }
    cout << "[Info] marker num = " << nums << ", save swc file to " << filePath << endl;
    swcFile.close();
    logFile.close();
    return true;
}

bool Neuron_Reconstruction :: adjustFinalNeuronTree(const QString &swcPath){
    string filepath = swcPath.toStdString();
    ifstream ifs(filepath);
    if (ifs.fail()) 
    {
        cout << "[Info] Open " << filepath << " failed." << endl;
        return false;
    }
    //Read the reconstructed .swc file
    NodeList nt;
    unordered_map<int, NeuronNode *> marker_map;
    unordered_map<NeuronNode *, int> parid_map;

    NeuronNode *soma_node;
    double rmax = 0;

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
        if (rmax < pNode->radius)
        {
            rmax = pNode->radius;
            soma_node = pNode;
        }
    }
    ifs.close();

    for (NeuronNode *pNode: nt) 
    {
        int parid = parid_map[pNode];
        if (parid == -1) continue;
        pNode->parent = marker_map[parid];
        pNode->parent->children.push_back(pNode);
    }

    //set the root point to soma node
    if (soma_node->parent != nullptr && !setNewRoot(soma_node)) return false;

    //delete the branch without connect soma
    NodeList tinyBranchBack, tinyBranchFront;
    for (NeuronNode *pNode: nt) 
    {   NeuronNode *branchPoint = pNode;
        if (!branchPoint->children.empty()) continue;
        while (branchPoint->parent !=nullptr)
            branchPoint = branchPoint->parent;
        if (branchPoint == soma_node) continue;
        tinyBranchBack.push_back(pNode);
        tinyBranchFront.push_back(branchPoint);
    }
    for (auto front: tinyBranchFront) 
    {
        if (front->parent == nullptr) continue;
        front->parent->children.removeOne(front);
        front->parent = nullptr;
    }
    for (auto back: tinyBranchBack) 
    {
        NeuronNode *next = nullptr;
        do {
            next = back->parent;
            delete back;
            back = next;
        } while (next != nullptr);
    }
    //delete the branch length less than 3
    //while(!pruneFinalBranch(min_branch_length,&nt));
    if(!pruneFinalBranch(min_branch_length,&nt))
    {
        cout << "[ERROR]: pruneFinalBranch failed." << endl;
        return false;   
    }
    //save swc file
    string savepath = filepath + "_prune.swc";
    ofstream swcFile(savepath, ios::binary);
    if (swcFile.fail()) 
    {
        cout << "[ERROR]: open save file failed." << endl;
        return false;
    }
    V3DLONG nums = 0;
    unordered_map<NeuronNode *, V3DLONG> ind;
    for (NeuronNode *node: nt) ind.insert(pair<NeuronNode *, V3DLONG>(node, ++nums));
    swcFile << "# marker path " << qPrintable(markerPath) << endl;
    swcFile << "# name " << filepath << endl;
    swcFile << "# n, type, x, y, z, radius, parent" << endl;
    nums = 0;
    for (NeuronNode *node: nt) {
        V3DLONG parent_id = (node->parent && ind.count(node->parent)) ? ind[node->parent] : -1;
        swcFile << ++nums << " " << node->type << " " << node->x << " " << node->y << " " << node->z << " " << node->radius
            << " " << parent_id << endl;
    }
    cout << "[Info] prune swc marker num = " << nums << endl;
    swcFile.close();
    return TRUE;
}

bool Neuron_Reconstruction :: pruneFinalBranch(int length, NodeList *NTree) {
    NodeList tinyBranchBack, tinyBranchFront;
    if (NTree == nullptr)  return 0;
    for (auto node: *(NTree)) 
    {
        if (!node->children.empty()) continue;
        NeuronNode *branchPoint = node;
        int count = 0;
        while (branchPoint->parent && branchPoint->parent->children.size() < 2 && count < length) 
        {
            branchPoint = branchPoint->parent;
            count++;
        }
        if (count < length && branchPoint->radius<20) // && branchPoint->parent
        {
            tinyBranchBack.push_back(node);
            tinyBranchFront.push_back(branchPoint);
        }
    }

    for (auto front: tinyBranchFront) 
    {
        if (front->parent == nullptr) continue;
        front->parent->children.removeOne(front);
        front->parent = nullptr;
    }
    for (auto back: tinyBranchBack) 
    {
        NeuronNode *next = nullptr;
        do {
            next = back->parent;
            delete back;
            back = next;
        } while (next != nullptr);
    }
    cout<< "[Info] Final swc pruned "<<tinyBranchFront.size()<<endl;
    return true;
    //if (tinyBranchFront.size()!=0) return true;
    //else return false;
}

bool block_segmentation(V3DPluginCallback2 &callback, V3DPluginArgList &args, string input_image, const QString method){
    QString pythonEnvironment((static_cast<vector<char *> *>(args.back().p))->at(0)),
    configurationPath((static_cast<vector<char *> *>(args.back().p))->at(1)),
    outputFolderPath((static_cast<vector<char *> *>(args.back().p))->at(2));

    QString inputPath = QString::fromStdString(input_image);
    //cout<<"pythonEnvironment: "<<pythonEnvironment.toStdString()<<endl;
    //cout<<"image file path: "<<inputPath.toStdString()<<endl;
    //cout<<"configurationPath"<<configurationPath.toStdString()<<endl;
    //cout<<"outputFolderPath"<<outputFolderPath.toStdString()<<endl;
    
    QString codepath = "F:\\neuron_reconstruction_system\\D_LSNARS\\singel_neuron_reconstruction\\src\\python\\block_segment.py";
    QString command = QString("conda activate %1 && python ");
    command.append(QString(codepath)).append(QString(" -i %2 -c %3 -o %4"));
#ifdef _WIN32
    int ret = system(qPrintable(command.arg(pythonEnvironment).arg(inputPath).arg(configurationPath).arg(outputFolderPath)));
#endif
    if (ret!=0) return FALSE;
    return TRUE;               
}

bool SPE_DNR(V3DPluginCallback2 &callback, V3DPluginArgList &args, bool isStartBlock){
    QString pythonEnvironment((static_cast<vector<char *> *>(args.back().p))->at(0)),
    inputPath((static_cast<vector<char *> *>(args.front().p))->at(0)),
    configurationPath((static_cast<vector<char *> *>(args.back().p))->at(1)),
    outputFolderPath((static_cast<vector<char *> *>(args.back().p))->at(2));
    //cout<<"pythonEnvironment: "<<pythonEnvironment.toStdString()<<endl;
    //cout<<"image file path: "<<inputPath.toStdString()<<endl;
    //cout<<"configurationPath"<<configurationPath.toStdString()<<endl;
    //cout<<"outputFolderPath"<<outputFolderPath.toStdString()<<endl;
    QString isStart;
    if(isStartBlock)  isStart.append("1");
    else isStart.append("0");
    
    QString codepath = "F:\\neuron_reconstruction_system\\D_LSNARS\\singel_neuron_reconstruction\\src\\python\\SPE_DNR.py";
    QString command = QString("conda activate %1 && python ");
    command.append(QString(codepath)).append(QString(" -i %2 -c %3 -s %4"));
#ifdef _WIN32
    int ret = system(qPrintable(command.arg(pythonEnvironment).arg(inputPath).arg(configurationPath).arg(isStart)));
#endif
    if (ret!=0) return FALSE;
    
    return TRUE;               
}

bool error_reconstruct(V3DPluginArgList &args, string input_path){
    QString pythonEnvironment((static_cast<vector<char *> *>(args.back().p))->at(0));
    QString swc_path = QString::fromStdString(input_path);
    QString codepath = "F:/neuron_reconstruction_system/D_LSNARS/singel_neuron_reconstruction/src/python/check_reconstruct.py";
    QString command = QString("conda activate %1 && python ");
    command.append(QString(codepath)).append(QString(" -i %2"));
#ifdef _WIN32
    int ret = system(qPrintable(command.arg(pythonEnvironment).arg(swc_path)));
#endif
    if (ret!=0) return FALSE;
    return TRUE;
}

bool resample(V3DPluginCallback2 &callback, const string &swcfile, string stepLen, const string &resampledSwcFile) {
    V3DPluginArgList inputArgsList, outputArgsList;
    vector<char *> infile{const_cast<char *>(swcfile.data())};
    vector<char *> inpara{const_cast<char *>(stepLen.data())};
    vector<char *> output{const_cast<char *>(resampledSwcFile.data())};
    inputArgsList.push_back(V3DPluginArgItem());
    inputArgsList.push_back(V3DPluginArgItem());
    inputArgsList.front().p = &infile;
    inputArgsList.back().p = &inpara;
    outputArgsList.push_back(V3DPluginArgItem());
    outputArgsList.front().p = &output;
    return callback.callPluginFunc("resample_swc", "resample_swc", inputArgsList, outputArgsList);
}


/*
bool whole_brain_soma_location(V3DPluginCallback2 &callback, V3DPluginArgList &args){
    QString pythonEnvironment((static_cast<vector<char *> *>(args.front().p))->at(0)),
    inputPath((static_cast<vector<char *> *>(args.front().p))->at(1)),
    configurationPath((static_cast<vector<char *> *>(args.front().p))->at(2)),
    outputFolderPath((static_cast<vector<char *> *>(args.front().p))->at(3));
    //cout<<"pythonEnvironment: "<<pythonEnvironment.toStdString()<<endl;
    //cout<<"image file path: "<<inputPath.toStdString()<<endl;
    //cout<<"configurationPath"<<configurationPath.toStdString()<<endl;
    //cout<<"outputFolderPath"<<outputFolderPath.toStdString()<<endl;

    QString codepath = "F:\\neuron_reconstruction_system\\D_LSNARS\\whole_brain_somas_detection\\src\\python\\Soma_location.py";
    QString command = QString("conda activate %1 && python ");
    command.append(QString(codepath)).append(QString(" -i %2 -c %3 -o %4"));
#ifdef _WIN32
    int ret = system(qPrintable(command.arg(pythonEnvironment).arg(inputPath).arg(configurationPath).arg(outputFolderPath)));
#endif
    if (ret!=0) return FALSE;
    return TRUE;               
}
*/
/*
bool resample(const string &swcfile, string stepLen, const string &resampledSwcFile){
    NodeList Ntree;
    readSWCtoNodeList(swcfile,Ntree);

    return true;
}
*/
char* GetCWD() {
	char* str;
	if ((str= getcwd(NULL, 0)) == NULL) {
		perror("getcwd error");
		return NULL;
	}
	else {
		printf("dll path %s\n", str);
		return str;
		free(str);
	}
}
