#include "soma_detection.h"

Soma_Detection :: Soma_Detection(V3DPluginCallback2& cb):
callback(cb)
{   
    cout<<"soma_detection start"<<endl;
}

void Soma_Detection :: WHOLE_BRAIM_SOMA_DETECTION() {
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
    
    //path to save method parameteras 
    configurationPath = outputFolderPath + "/configuration.yaml";
    saveConfiguration(configurationPath);
    
    //Calculate program runtime
    QElapsedTimer timer{};
    timer.start(); 

    cout<<"Candidate block screening"<<endl;
    if(!Candidate_block_screening(imagePath,256))
    {
        //allTargetList.clear();
        cout<<"Candidate block screening ERROW"<<endl;
    }
    
    cout<<"Soma_location "<<endl;
    if(!Soma_location(imagePath,256))
    {
        cout<<"Soma Detection ERROW"<<endl;
    }
    
    qint64 endTime = timer.nsecsElapsed()/1e9/3600;
    cout<<"cost time: "<<endTime<<" h"<<endl;
    return ;
}

bool Soma_Detection :: Candidate_block_screening(const QString &image_path, int blockSize){
    //set parameters and extract image blocks for soma_segment 
    V3DPluginArgList ArgsList;
    ArgsList.push_back(V3DPluginArgItem());
    vector<char *>  paras;
    ArgsList.back().p = &paras;
    
    parasString.emplace_back(pytorchPath.toStdString());
    parasString.emplace_back(tempFolderPath.toStdString());
    parasString.emplace_back(configurationPath.toStdString());
    parasString.emplace_back(outputFolderPath.toStdString());
    for (auto &para: parasString) {
        paras.push_back(const_cast<char *>(para.data()));
    }
    
    //read all image blocks path
    ObtainALLimagePATH(image_path);
    cout<<"image block num: "<<imagepathlist.size()<<endl;
    int batch_num = 0;
    QDir temppath(tempFolderPath);
    //candidate block screening
    
    for (int i = 0; i<imagepathlist.size();i++)
    {
        Image4DSimple *img = new Image4DSimple();
        if(!getVolume1(callback, const_cast<char *>(imagepathlist[i].data()),*img)){
            cout<<"read failed: "<<imagepathlist[i].data()<<endl;
            return FALSE;
            }
        
        if(MPL(img)){
            string imagename = imagepathlist[i];
            int index = imagename.rfind('\\');
            int length = imagename.size()-index;
            imagename = imagename.substr(index,length);
            string tempimagesavepath = tempFolderPath.toStdString()+imagename;

            callback.saveImage(img, const_cast<char *>(tempimagesavepath.data()));
            batch_num++;
            
            if(batch_num == batchsize || i == imagepathlist.size()-1 ){
                if(!whole_brain_candidate_block_screening(callback, ArgsList)){
                        cout<<"candidate_block_screening python code run errow"<<endl;
                        return FALSE;
                }
                
                cout<<"Screening progress: "<<i+1<<"/"<<imagepathlist.size()<<endl;
                QStringList slist=temppath.entryList(temppath.filter()|QDir::NoDotAndDotDot);
                for (int j = 0; j < slist.size(); j++)
                {
                    temppath.remove(slist[j]);
                }
                batch_num = 0;
                //break;
            } 
        }
        //img.cleanExistData();
        delete img;
    }
    
    return TRUE;
}

bool Soma_Detection :: saveConfiguration(const QString &savePath) {
    QFile outFile(savePath);
    if (outFile.exists()) outFile.remove();
    if (!outFile.open(QFile::ReadWrite)) return false;
    QTextStream textStream(&outFile);
    textStream << "CE: "<< CE_value<<endl;
    textStream << "SMA: "<< SMA_value<<endl;
    textStream << "DB_eps: "<< DB_eps_value<<endl;
    textStream << "DB_MP: "<< DB_MP_value<<endl;
    //textStream << "OutputPath: " << QFileInfo(savePath).absolutePath() << endl;
    outFile.close();
    return true;
}

bool Soma_Detection :: ObtainALLimagePATH(const QString &image_path){
    string imagefile_path = image_path.toStdString();    
    vector<string> x_temp,y_temp,z_temp;
	getAllFiles(imagefile_path, x_temp,"");
	for (int i = 1; i < x_temp.size();++i	)
	{
        //cout<<x_temp[i]<<endl;
        if (x_temp[i].find('.')!=-1) continue;
        getAllFiles(x_temp[i],y_temp,"");
        for(int j=1;j<y_temp.size();++j)
		{   
            if (y_temp[j].find('.')!=-1) continue;
            getAllFiles(y_temp[j],z_temp,".tif");
            for (int k=1; k<z_temp.size(); ++k)
            {
               imagepathlist.push_back(z_temp[k]);
            }
            z_temp.clear();
        }
        y_temp.clear();
	}
	return TRUE;
}

bool Soma_Detection :: Soma_location(const QString &image_path, int blockSize){
    V3DPluginArgList ArgsList;
    ArgsList.push_back(V3DPluginArgItem());
    vector<char *>  paras;
    ArgsList.back().p = &paras;
    parasString.clear();
    parasString.emplace_back(pytorchPath.toStdString());
    parasString.emplace_back(image_path.toStdString());
    parasString.emplace_back(configurationPath.toStdString());
    parasString.emplace_back(outputFolderPath.toStdString());
    for (auto &para: parasString) {
        paras.push_back(const_cast<char *>(para.data()));
    }
    
    // bulid soma_block marker save files
    QDir outputDir(outputFolderPath);
    outputDir.mkpath("soma_blocks");

    if(!whole_brain_soma_location(callback,ArgsList)){
        cout<<"soma_location python code run errow"<<endl;
        return FALSE;
    }
    return TRUE;
}

bool whole_brain_candidate_block_screening(V3DPluginCallback2 &callback, V3DPluginArgList &args){
    QString pythonEnvironment((static_cast<vector<char *> *>(args.front().p))->at(0)),
    inputPath((static_cast<vector<char *> *>(args.front().p))->at(1)),
    configurationPath((static_cast<vector<char *> *>(args.front().p))->at(2)),
    outputFolderPath((static_cast<vector<char *> *>(args.front().p))->at(3));
    //cout<<"pythonEnvironment: "<<pythonEnvironment.toStdString()<<endl;
    //cout<<"image file path: "<<inputPath.toStdString()<<endl;
    //cout<<"configurationPath"<<configurationPath.toStdString()<<endl;
    //cout<<"outputFolderPath"<<outputFolderPath.toStdString()<<endl;
    QString codepath = "F:\\neuron_reconstruction_system\\D_LSNARS\\whole_brain_somas_detection\\src\\python\\Candidate_Block_Screening.py";
    QString command = QString("conda activate %1 && python ").append(QString(codepath));
    command.append(QString(" -i %2 -c %3 -o %4"));     

#ifdef _WIN32
    int ret = system(qPrintable(command.arg(pythonEnvironment).arg(inputPath).arg(configurationPath).arg(outputFolderPath)));
#endif
    if (ret!=0) return FALSE;
    return TRUE;
}

/*
string whole_brain_candidate_block_screening(V3DPluginCallback2 &callback, V3DPluginArgList &args, QProcess &process) {
    QString pythonEnvironment((static_cast<vector<char *> *>(args.front().p))->at(0)),
            inputPath((static_cast<vector<char *> *>(args.front().p))->at(1)),
            configurationPath((static_cast<vector<char *> *>(args.front().p))->at(2)),
            outputFolderPath((static_cast<vector<char *> *>(args.front().p))->at(3));
    cout<<"python start"<<endl;
    QString codepath = "F:\\neuron_reconstruction_system\\D_LSNARS\\whole_brain_somas_detection\\src\\python\\Candidate_Block_Screening.py";
    pythonEnvironment.append("/Python.exe");
    cout<<pythonEnvironment.toStdString()<<endl;

    QStringList command;
    command.append(codepath);
    command.append("-i");
    command.append(inputPath);
    command.append("-c");
    command.append(configurationPath);
    command.append("-o");
    command.append(outputFolderPath);
#ifdef _WIN32
    
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.start(pythonEnvironment, command);
    //process->waitForStarted();
    
    process.waitForFinished(); // Wait indefinitely for process to finish
    int ret = process.exitCode();
    QByteArray error = process.readAllStandardError();
    QByteArray output = process.readAllStandardOutput();
    qDebug() << "Python output: " << output;

    QProcess::ExitStatus exitStatus = process.exitStatus();
    if (ret != 0 || exitStatus != QProcess::NormalExit) {
        // 处理错误情况，可以记录错误信息
        qDebug() << "Python script execution failed. Exit code:" << ret;
        qDebug() << "Error:" << error;
        return {};
    } 
    process.close();
    process.deleteLater();
    return outputFolderPath.toStdString();
#endif
}
*/
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
