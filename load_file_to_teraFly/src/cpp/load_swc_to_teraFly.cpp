#include "load_swc_to_teraFly.h"

LoadSwcTeraFlyDialog::LoadSwcTeraFlyDialog(V3DPluginCallback2 &callback, QWidget *parent) :
        _callback(callback),
        input_swc_path_label1(new QLabel("input .swc file1: ")),
        input_swc_path_text1(new QLineEdit()),
        input_swc_path_label2(new QLabel("input .swc file2: ")),
        input_swc_path_text2(new QLineEdit()),
        offsetX(new QLabel("OriginX: ")),
        offsetY(new QLabel("OriginY: ")),
        offsetZ(new QLabel("OriginZ: ")),
        offsetXbox(new QSpinBox()),
        offsetYbox(new QSpinBox()),
        offsetZbox(new QSpinBox()),
        selete_button1(new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "")),
		selete_button2(new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "")),
        loadSwc1(new QPushButton("load swc1")),
		loadSwc2(new QPushButton("load swc2")),
        extractMarker(new QPushButton("extract marker")),
        hbox1(new QHBoxLayout()),
        hbox2(new QHBoxLayout()),
        hbox3(new QHBoxLayout()),
		hbox4(new QHBoxLayout()),
		hbox5(new QHBoxLayout()),
        layout(new QGridLayout()) 
        {
    this->setParent(parent);
    offsetXbox->setValue(0);
    offsetXbox->setRange(0, INT_MAX);
    offsetYbox->setValue(0);
    offsetYbox->setRange(0, INT_MAX);
    offsetZbox->setValue(0);
    offsetZbox->setRange(0, INT_MAX);

    hbox1->addWidget(input_swc_path_label1);
    hbox1->addWidget(input_swc_path_text1);
    hbox1->addWidget(selete_button1);

    hbox2->addWidget(offsetX);
    hbox2->addWidget(offsetXbox);
    hbox2->addWidget(offsetY);
    hbox2->addWidget(offsetYbox);
    hbox2->addWidget(offsetZ);
    hbox2->addWidget(offsetZbox);

	//hbox3->addWidget(extractMarker);
	hbox3->addWidget(loadSwc1);

	hbox4->addWidget(input_swc_path_label2);
	hbox4->addWidget(input_swc_path_text2);
	hbox4->addWidget(selete_button2);

	hbox5->addWidget(loadSwc2);

    layout->addLayout(hbox1, 0, 0);
    layout->addLayout(hbox2, 1, 0);
	layout->addLayout(hbox3, 2, 0);
    layout->addLayout(hbox4, 3, 0);
	layout->addLayout(hbox5, 4, 0);
    setLayout(layout);
    connect(selete_button1, SIGNAL(clicked()), this, SLOT(select_swc1()));
	connect(selete_button2, SIGNAL(clicked()), this, SLOT(select_swc2()));

    connect(input_swc_path_text1, SIGNAL(editingFinished()), this, SLOT(update()));
	connect(input_swc_path_text2, SIGNAL(editingFinished()), this, SLOT(update()));

    connect(offsetXbox, SIGNAL(valueChanged(int)), this, SLOT(update()));
    connect(offsetYbox, SIGNAL(valueChanged(int)), this, SLOT(update()));
    connect(offsetZbox, SIGNAL(valueChanged(int)), this, SLOT(update()));

    connect(loadSwc1, SIGNAL(clicked()), this, SLOT(load1()));
	connect(loadSwc2, SIGNAL(clicked()), this, SLOT(load2()));

    //connect(extractMarker, SIGNAL(clicked()), this, SLOT(extract()));
}


LoadSwcTeraFlyDialog::~LoadSwcTeraFlyDialog() {
    delete layout;
}

void LoadSwcTeraFlyDialog::update() {
    input_swc_path1 = input_swc_path_text1->text();
	input_swc_path2 = input_swc_path_text2->text();
}

void LoadSwcTeraFlyDialog::select_swc1() {
    QString select_swc_file = QFileDialog::getOpenFileName(this, "select the .swc file.", "", "*.swc *.eswc");
    if (select_swc_file.isEmpty()) return;
	input_swc_path1 = select_swc_file;
	input_swc_path_text1->setText(select_swc_file);
	
}

void LoadSwcTeraFlyDialog::select_swc2() {
	QString select_swc_file = QFileDialog::getOpenFileName(this, "select the .swc file.", "", "*.swc *.eswc");
	if (select_swc_file.isEmpty()) return;
	input_swc_path2 = select_swc_file;
	input_swc_path_text2->setText(select_swc_file);
}

void LoadSwcTeraFlyDialog::load1(){
    QString TeraFly_name = _callback.getPathTeraFly();
    if (TeraFly_name.isEmpty()) return;
    NeuronTree nt1 = readSWC_file(input_swc_path1);
	if (offsetXbox->value() || offsetYbox->value() || offsetZbox->value()) {
		V3DLONG OriginX, OriginY, OriginZ;
		OriginX = offsetXbox->value();
		OriginY = offsetYbox->value();
		OriginZ = offsetZbox->value();
		for (NeuronSWC &n : nt1.listNeuron) {
			n.x += OriginX;
			n.y += OriginY;
			n.z += OriginZ;
		}
	}
	_callback.setSWCTeraFly(nt1);
}

void LoadSwcTeraFlyDialog::load2(){
	QString TeraFly_name = _callback.getPathTeraFly();
	if (TeraFly_name.isEmpty()) return;
	NeuronTree nt2 = readSWC_file(input_swc_path2);
	if (offsetXbox->value() || offsetYbox->value() || offsetZbox->value()) {
		V3DLONG OriginX, OriginY, OriginZ;
		for (NeuronSWC &n : nt2.listNeuron) {
			n.x += 0;
			n.y += 0;
			n.z += 0;
		}
	}
	_callback.setSWCTeraFly(nt2);
}