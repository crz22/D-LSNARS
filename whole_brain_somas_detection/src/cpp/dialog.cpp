#include "dialog.h"

Detection_Dialog ::Detection_Dialog(V3DPluginCallback2 &callback, QWidget *parent):
    _callback(callback),
    OverallLayout(new QGridLayout()),
    groupBox_filepath(new QGroupBox("File path",this)),
    groupBox_detection_parameters(new QGroupBox("Parameters Set",this))
    //groupBox_segmentation(new QGroupBox("Neuron segmentation",this)),
    //groupBox_reconstruction(new QGroupBox("Neuron reconstruction",this)),
    //groupBox_check(new QGroupBox("Neuron check",this))
    //ui(new Ui::NeuronReconstructorDialog)
{
    //window set
    this->setWindowTitle("whole brain soma detection");
    this->resize(400,100);
    this->setWindowFlags(windowFlags() | Qt::WindowMaximizeButtonHint);
    this->setWindowFlags(windowFlags() | Qt::WindowMinimizeButtonHint);
    
    //Set file path
    Layout_filepath = new QGridLayout();

    HBox_input_path = new QHBoxLayout();
    input_path_label = new QLabel("TeraFly file path: ");
    input_path_text = new QLineEdit();
    input_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    HBox_input_path->addWidget(input_path_label);
    HBox_input_path->addWidget(input_path_text);
    HBox_input_path->addWidget(input_path_button);

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

    HBox_pytorch_path = new QHBoxLayout();
    pytorch_path_label = new QLabel("Pytorch path: ");
    pytorch_path_text = new QLineEdit();
    pytorch_path_text->setText("D:/ruanjian/minicoda/envs/pytorch");
    pytorch_path_value = pytorch_path_text->text();
    pytorch_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    HBox_pytorch_path->addWidget(pytorch_path_label);
    HBox_pytorch_path->addWidget(pytorch_path_text);
    HBox_pytorch_path->addWidget(pytorch_path_button);
    
    Layout_filepath->addLayout(HBox_input_path,0,0);
    Layout_filepath->addLayout(HBox_save_path,1,0);
    Layout_filepath->addLayout(HBox_pytorch_path,2,0);
    groupBox_filepath->setLayout(Layout_filepath);

    connect(input_path_button, SIGNAL(clicked()), this, SLOT(Select_input_path()));
    connect(save_path_button, SIGNAL(clicked()), this, SLOT(Select_save_path()));
    connect(pytorch_path_button, SIGNAL(clicked()), this, SLOT(Select_pytorch_path()));
    connect(save_path_text, SIGNAL(editingFinished()), this, SLOT(update_save_path()));
    connect(input_path_text, SIGNAL(editingFinished()), this, SLOT(update_input_path()));
    connect(pytorch_path_text, SIGNAL(editingFinished()), this, SLOT(update_pytorch_path()));
    
    //Set soma detection parameters
    Layout_detection_parameters = new QGridLayout();
    
    CE = new QDoubleSpinBox();
    CE->setValue(1);
    CE_value = CE->value();
    CE_label = new QLabel("CE: ");
    HBox_CE = new QHBoxLayout();
    HBox_CE->addWidget(CE_label);
    HBox_CE->setSpacing(10);
    HBox_CE->addWidget(CE);
    HBox_CE->setMargin(10);
    connect(CE,SIGNAL(editingFinished()),this,SLOT(update_CE()));

    SMA = new QSpinBox(); 
    SMA->setValue(35);
    SMA_value = SMA->value();
    SMA_label = new QLabel("SMA: ");
    HBox_SMA = new QHBoxLayout();
    HBox_SMA->addWidget(SMA_label);
    HBox_SMA->setSpacing(10);
    HBox_SMA->addWidget(SMA);
    HBox_SMA->setMargin(10);
    connect(SMA,SIGNAL(editingFinished()),this,SLOT(update_SMA()));

    DB_eps = new QSpinBox();
    DB_eps->setMaximum(1024);
    DB_eps->setValue(256);
    DB_eps_value = DB_eps->value();
    DB_eps_label = new QLabel("DB_eps: ");
    HBox_DB_eps = new QHBoxLayout();
    HBox_DB_eps->addWidget(DB_eps_label);
    HBox_DB_eps->setSpacing(10);
    HBox_DB_eps->addWidget(DB_eps);
    HBox_DB_eps->setMargin(10);
    connect(DB_eps,SIGNAL(editingFinished()),this,SLOT(update_DB_eps()));

    DB_MP = new QSpinBox();
    DB_MP->setValue(50);
    DB_MP_value = DB_MP->value();
    DB_MP_label = new QLabel("DB_MP: ");
    HBox_DB_MP = new QHBoxLayout();
    HBox_DB_MP->addWidget(DB_MP_label);
    HBox_DB_MP->setSpacing(10);
    HBox_DB_MP->addWidget(DB_MP);
    HBox_DB_MP->setMargin(10);
    connect(DB_MP,SIGNAL(editingFinished()),this,SLOT(update_DB_MP()));

    Layout_detection_parameters->addLayout(HBox_CE,0,0);
    //Layout_detection_parameters->setSpacing(50);
    Layout_detection_parameters->addLayout(HBox_SMA,0,1);
    Layout_detection_parameters->addLayout(HBox_DB_eps,1,0);
    //Layout_detection_parameters->setSpacing(50);
    Layout_detection_parameters->addLayout(HBox_DB_MP,1,1);
    groupBox_detection_parameters->setLayout(Layout_detection_parameters);

    

    Start_button = new QPushButton("Start");
    connect(Start_button,SIGNAL(clicked()),this,SLOT(Start_soma_detection()));
    //OverallLayout
    OverallLayout->addWidget(groupBox_filepath,0,0);
    OverallLayout->addWidget(groupBox_detection_parameters,1,0);
    OverallLayout->addWidget(Start_button,2,0);

    this->setLayout(OverallLayout);

    cout<<"initial finish"<<endl;
}

Detection_Dialog :: ~Detection_Dialog(){
    cout<<" "<<endl;
}

void Detection_Dialog :: Select_input_path(){
    QString select_input_file = QFileDialog::getExistingDirectory(this, "select the TeraFly file.", "/");
    if (select_input_file.isEmpty()) return;
	input_path_value = select_input_file;
	input_path_text->setText(select_input_file);
    cout<<"input_path: "<<input_path_value.toStdString()<<endl;
}

void Detection_Dialog :: Select_save_path(){
    QString select_save_file = QFileDialog::getExistingDirectory(this, "select the save file.", "/");
    if (select_save_file.isEmpty()) return;
	save_path_value = select_save_file;
	save_path_text->setText(select_save_file);
    cout<<"save_path: "<<save_path_value.toStdString()<<endl;
}

void Detection_Dialog :: Select_pytorch_path(){
    QString select_pytorch_file = QFileDialog::getExistingDirectory(this, "select the pytorch file.", "/");
    if (select_pytorch_file.isEmpty()) return;
	pytorch_path_value = select_pytorch_file;
	pytorch_path_text->setText(select_pytorch_file);
    cout<<"pytorch_path: "<<pytorch_path_value.toStdString()<<endl;
}

void Detection_Dialog :: update_save_path(){
    save_path_value = save_path_text->text();
    cout<<"save_path: "<<save_path_value.toStdString()<<endl;
}

void Detection_Dialog :: update_input_path(){
    input_path_value = input_path_text->text();
    cout<<"input_path: "<<input_path_value.toStdString()<<endl;
}

void Detection_Dialog :: update_pytorch_path(){
    pytorch_path_value = pytorch_path_text->text();
    cout<<"pytorch_path: "<<pytorch_path_value.toStdString()<<endl;
}

void Detection_Dialog :: update_CE(){
    CE_value = CE->value();
    cout<<"CE:  "<<CE_value<<endl;
}

void Detection_Dialog :: update_SMA(){
    SMA_value = SMA->value();
    cout<<"SMA:  "<<SMA_value<<endl;
}

void Detection_Dialog :: update_DB_eps(){
    DB_eps_value = DB_eps->value();
    cout<<"DB_esp:  "<<DB_eps_value<<endl;
}

void Detection_Dialog :: update_DB_MP(){
    DB_MP_value = DB_MP->value();
    cout<<"CE:  "<<DB_MP_value<<endl;
}


void Detection_Dialog :: Start_soma_detection(){
    soma_detection = new Soma_Detection(_callback);
    soma_detection->imagePath = input_path_value;
    soma_detection->outputFolderPath = save_path_value;
    soma_detection->pytorchPath = pytorch_path_value;
    soma_detection->CE_value = CE_value;
    soma_detection->SMA_value = SMA_value;
    soma_detection->DB_eps_value = DB_eps_value;
    soma_detection->DB_MP_value = DB_MP_value;
    
    soma_detection->WHOLE_BRAIM_SOMA_DETECTION();
    cout<<"finished"<<endl;
}

