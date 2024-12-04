#include "dialog.h"

Reconstruction_Dialog ::Reconstruction_Dialog(V3DPluginCallback2 &callback, QWidget *parent):
     _callback(callback),
    OverallLayout(new QGridLayout()),
    groupBox_filepath(new QGroupBox("File path",this)),
    traceMethod(new QComboBox (this)),
    segmentMethod(new QComboBox (this)),
    groupBox_reconstruct_parameters(new QGroupBox("parameter set",this))
{
    //window set
    this->setWindowTitle("singel neuron reconstruction");
    this->resize(400,100);
    this->setWindowFlags(windowFlags() | Qt::WindowMaximizeButtonHint);
    this->setWindowFlags(windowFlags() | Qt::WindowMinimizeButtonHint);

    //Set file path
    SETfilepath(callback);
    
    //Set neuron reconstruct parameters
    SETparameter();

    /**/
    Start_button = new QPushButton("Start");
    connect(Start_button,SIGNAL(clicked()),this,SLOT(Start_neuron_reconstruction()));

    OverallLayout->addWidget(groupBox_filepath,0,0);
    OverallLayout->addWidget(groupBox_reconstruct_parameters,1,0);
    OverallLayout->addWidget(Start_button,2,0);

    this->setLayout(OverallLayout);

    cout<<"initial finish"<<endl;

}

void Reconstruction_Dialog :: SETfilepath(V3DPluginCallback2& callback){
    //Set file path
    Layout_filepath = new QGridLayout();

    HBox_input_path = new QHBoxLayout();
    input_path_label = new QLabel("TeraFly file path: ");
    input_path_text = new QLineEdit();
    input_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    v3dhandle curWin = callback.currentImageWindow();
    if (curWin) {
        QString imageName = callback.getImageName(curWin);
        if (!imageName.endsWith(".tif")) {
            input_path_text->setText(callback.getPathTeraFly());
            //teraflyChecker->setChecked(Qt::CheckState::Checked);
        } 
        else input_path_text->setText(imageName);
        input_path_value = input_path_text->text();
        input_path_text->setEnabled(false);
        input_path_button->setEnabled(false);
    }

    HBox_input_path->addWidget(input_path_label);
    HBox_input_path->addWidget(input_path_text);
    HBox_input_path->addWidget(input_path_button);
    
    /**/
    HBox_marker_path = new QHBoxLayout();
    marker_path_label = new QLabel("Marker file path: ");
    marker_path_text = new QLineEdit();
    marker_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    HBox_marker_path->addWidget(marker_path_label);
    HBox_marker_path->addWidget(marker_path_text);
    HBox_marker_path->addWidget(marker_path_button);

    /**/
    HBox_save_path = new QHBoxLayout();
    save_path_label = new QLabel("Save file path:    ");
    save_path_text = new QLineEdit();
    save_path_text->setText("F:/neuron_reconstruction_system/test/reconstruct_result");
    save_path_value = save_path_text->text();
    save_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    HBox_save_path->addWidget(save_path_label);
    HBox_save_path->addWidget(save_path_text);
    HBox_save_path->addWidget(save_path_button);
    
    /**/
    HBox_pytorch_path = new QHBoxLayout();
    pytorch_path_label = new QLabel("Pytorch path: ");
    pytorch_path_text = new QLineEdit();
    pytorch_path_text->setText("D:/ruanjian/minicoda/envs/pytorch");
    pytorch_path_value = pytorch_path_text->text();
    pytorch_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    HBox_pytorch_path->addWidget(pytorch_path_label);
    HBox_pytorch_path->addWidget(pytorch_path_text);
    HBox_pytorch_path->addWidget(pytorch_path_button);
    
    /**/
    Layout_filepath->addLayout(HBox_input_path,0,0);
    Layout_filepath->addLayout(HBox_marker_path,1,0);
    Layout_filepath->addLayout(HBox_save_path,2,0);
    Layout_filepath->addLayout(HBox_pytorch_path,3,0);
    groupBox_filepath->setLayout(Layout_filepath);

    connect(input_path_button, SIGNAL(clicked()), this, SLOT(Select_input_path()));
    connect(marker_path_button, SIGNAL(clicked()), this, SLOT(Select_marker_path()));
    connect(save_path_button, SIGNAL(clicked()), this, SLOT(Select_save_path()));
    connect(pytorch_path_button, SIGNAL(clicked()), this, SLOT(Select_pytorch_path()));

    connect(save_path_text, SIGNAL(editingFinished()), this, SLOT(update_save_path()));
    connect(input_path_text, SIGNAL(editingFinished()), this, SLOT(update_input_path()));
    connect(marker_path_text, SIGNAL(editingFinished()), this, SLOT(update_marker_path()));
    connect(pytorch_path_text, SIGNAL(editingFinished()), this, SLOT(update_pytorch_path()));
}

void Reconstruction_Dialog :: SETparameter(){
    //Set neuron reconstruct parameters
    Layout_reconstruct_parameters = new QGridLayout();
    HBox_Method = new QHBoxLayout();
    traceMethod_label = new QLabel("traceMethod: ");
    traceMethod->insertItem(0,"SPEDNR");
    traceMethod->insertItem(1,"APP2");
    traceMethod_value = "SPE_DNR";
    connect(traceMethod,SIGNAL(currentIndexChanged(int)),this,SLOT(SelecttraceMethod()));
    
    segmentMethod_label = new QLabel("segmentMethod: ");
    segmentMethod->insertItem(0,"DTANET");
    segmentMethod->insertItem(1,"UNET3D");
    segmentMethod_value = "DTANET";
    connect(segmentMethod,SIGNAL(currentIndexChanged(int)),this,SLOT(SelectsegmentMethod()));
    
    HBox_Method->addWidget(traceMethod_label);
    HBox_Method->addWidget(traceMethod);
    HBox_Method->addWidget(segmentMethod_label);
    HBox_Method->addWidget(segmentMethod);

    /**/
    HBox_blockSize = new QHBoxLayout();
    blockSize_label = new QLabel("blockSize: ");
    blockSize_input = new QSpinBox();
    blockSize_input->setMaximum(1024);
    blockSize_input->setButtonSymbols(QAbstractSpinBox:: NoButtons);
    blockSize_input->setValue(256);
    blockSize_value = blockSize_input->value();
    //save_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");
    connect(blockSize_input,SIGNAL(editingFinished()),this,SLOT(update_blockSize()));

    HBox_blockSize->addWidget(blockSize_label);
    HBox_blockSize->addWidget(blockSize_input);
    
    /**/
    HBox_margin_lamd = new QHBoxLayout();
    margin_lamd_label = new QLabel("margin_lamd: ");
    margin_lamd_input = new QDoubleSpinBox();
    margin_lamd_input->setMaximum(0.5);
    margin_lamd_input->setButtonSymbols(QAbstractSpinBox:: NoButtons);
    margin_lamd_input->setValue(0.05);
    margin_lamd_value = margin_lamd_input->value();
    //save_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");
    connect(margin_lamd_input,SIGNAL(editingFinished()),this,SLOT(update_margin_lamd()));

    HBox_margin_lamd->addWidget(margin_lamd_label);
    HBox_margin_lamd->addWidget(margin_lamd_input);

    /**/
    HBox_MaxpointNUM = new QHBoxLayout();
    MaxpointNUM_label = new QLabel("MaxpointNUM: ");
    MaxpointNUM_input = new QSpinBox();
    MaxpointNUM_input->setMaximum(1e5);
    MaxpointNUM_input->setButtonSymbols(QAbstractSpinBox:: NoButtons);
    MaxpointNUM_input->setValue(5e4);
    MaxpointNUM_value = MaxpointNUM_input->value();
    //save_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");
    connect(MaxpointNUM_input,SIGNAL(editingFinished()),this,SLOT(update_MaxpointNUM()));

    HBox_MaxpointNUM->addWidget(MaxpointNUM_label);
    HBox_MaxpointNUM->addWidget(MaxpointNUM_input);

    /**/
    HBox_min_branch_length = new QHBoxLayout();
    min_branch_length_label = new QLabel("min_branch_length: ");
    min_branch_length_input = new QSpinBox();
    min_branch_length_input->setMaximum(10);
    min_branch_length_input->setButtonSymbols(QAbstractSpinBox:: NoButtons);
    min_branch_length_input->setValue(3);
    min_branch_length_value = min_branch_length_input->value();
    //save_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");
    connect(min_branch_length_input,SIGNAL(editingFinished()),this,SLOT(update_min_branch_length()));
    
    HBox_min_branch_length->addWidget(min_branch_length_label);
    HBox_min_branch_length->addWidget(min_branch_length_input);
    
    /**/
    Layout_reconstruct_parameters->addLayout(HBox_Method,0,0,1,2);
    Layout_reconstruct_parameters->addLayout(HBox_blockSize,1,0);
    Layout_reconstruct_parameters->addLayout(HBox_margin_lamd,1,1);
    Layout_reconstruct_parameters->addLayout(HBox_MaxpointNUM,2,0);
    Layout_reconstruct_parameters->addLayout(HBox_min_branch_length,2,1);
    /**/
    groupBox_reconstruct_parameters->setLayout(Layout_reconstruct_parameters);
}

Reconstruction_Dialog ::~Reconstruction_Dialog(){
    cout<<"finish "<<endl;
}

void Reconstruction_Dialog :: Select_input_path(){
    QString select_input_file = QFileDialog::getExistingDirectory(this, "select the TeraFly file.", "/");
    if (select_input_file.isEmpty()) return;
	input_path_value = select_input_file;
	input_path_text->setText(select_input_file);
    cout<<"input_path: "<<input_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: Select_marker_path(){
    QString select_marker_file = QFileDialog::getOpenFileName(this, "select the .marker file.", "", "*.marker *.apo");
    if (select_marker_file.isEmpty()) return;
	marker_path_value = select_marker_file;
	marker_path_text->setText(select_marker_file);
    cout<<"marker_path: "<<marker_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: Select_save_path(){
    QString select_save_file = QFileDialog::getExistingDirectory(this, "select the save file.", "/");
    if (select_save_file.isEmpty()) return;
	save_path_value = select_save_file;
	save_path_text->setText(select_save_file);
    cout<<"save_path: "<<save_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: Select_pytorch_path(){
    QString select_pytorch_file = QFileDialog::getExistingDirectory(this, "select the pytorch file.", "/");
    if (select_pytorch_file.isEmpty()) return;
	pytorch_path_value = select_pytorch_file;
	pytorch_path_text->setText(select_pytorch_file);
    cout<<"pytorch_path: "<<pytorch_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: update_save_path(){
    save_path_value = save_path_text->text();
    cout<<"save_path: "<<save_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: update_input_path(){
    input_path_value = input_path_text->text();
    cout<<"input_path: "<<input_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: update_marker_path(){
    marker_path_value = marker_path_text->text();
    cout<<"marker_path: "<<marker_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: update_pytorch_path(){
    pytorch_path_value = pytorch_path_text->text();
    cout<<"pytorch_path: "<<pytorch_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: SelecttraceMethod(){
    int index = traceMethod->currentIndex();

    if (index == SPEDNR)
    {
        //function1->outputFolderPath = save_path1;
        //Soma_Detection(_callback);
        //function1 = new SomaDetection(_callback);
        //groupBox_soma_detection ->setEnabled(TRUE);
        traceMethod_value = "SPE_DNR";
        cout<<"traceMethod: SPE_DNR"<<endl;
    }
    else if(index == APP2)
    {
        cout<<"wait"<<endl;
    }
}

void Reconstruction_Dialog :: SelectsegmentMethod(){
    int index = segmentMethod->currentIndex();

    if (index == DTANET)
    {
        segmentMethod_value = "DTANET";
        cout<<"segmentMethod: DTANET"<<endl;
    }
    else if(index == UNET3D)
    {
        cout<<"wait"<<endl;
    }
}

void Reconstruction_Dialog :: update_blockSize(){
    blockSize_value = blockSize_input->value();
    cout<<"blockSize:  "<<blockSize_value<<endl;
}

void Reconstruction_Dialog :: update_margin_lamd(){
    margin_lamd_value = margin_lamd_input->value();
    cout<<"margin_lamd:  "<<margin_lamd_value<<endl;
}

void Reconstruction_Dialog :: update_MaxpointNUM(){
    MaxpointNUM_value = MaxpointNUM_input->value();
    cout<<"MaxpointNUM:  "<<MaxpointNUM_value<<endl;
}

void Reconstruction_Dialog :: update_min_branch_length(){
    min_branch_length_value = min_branch_length_input->value();
    cout<<"min_branch_length:  "<<min_branch_length_value<<endl;
}

void Reconstruction_Dialog :: Start_neuron_reconstruction(){
    
    neuron_reconstruction = new Neuron_Reconstruction(_callback);
    neuron_reconstruction->imagePath = input_path_value;
    neuron_reconstruction->markerPath = marker_path_value;
    neuron_reconstruction->outputFolderPath = save_path_value;
    neuron_reconstruction->pytorchPath = pytorch_path_value;

    neuron_reconstruction->traceMethod = traceMethod_value;
    neuron_reconstruction->segmentMethod = segmentMethod_value;
    neuron_reconstruction->blockSize = blockSize_value;
    neuron_reconstruction->margin_lamd = margin_lamd_value;
    neuron_reconstruction->MaxpointNUM = MaxpointNUM_value;
    neuron_reconstruction->min_branch_length = min_branch_length_value;

    neuron_reconstruction->node_step = 2;
    neuron_reconstruction->branch_MAXL = 1000;
    neuron_reconstruction->Angle_T = 1.57;  //1.047
    neuron_reconstruction->Lamd = 4;
    
    neuron_reconstruction->SINGEL_NEURON_RCONSTRUCT();
    delete neuron_reconstruction;
    cout<<"finished"<<endl;
}