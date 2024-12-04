#include "load_mark_to_teraFly.h"
using namespace std;

LoadMarkTeraFlyDialog::LoadMarkTeraFlyDialog(V3DPluginCallback2 &callback, QWidget *parent) :
        _callback(callback),
        input_mark_path_label1(new QLabel("input .mark file1: ")),
        input_mark_path_text1(new QLineEdit()),
        //input_swc_path_label2(new QLabel("input .swc file2: ")),
        //input_swc_path_text2(new QLineEdit()),
        offsetX(new QLabel("OriginX: ")),
        offsetY(new QLabel("OriginY: ")),
        offsetZ(new QLabel("OriginZ: ")),
        offsetXbox(new QSpinBox()),
        offsetYbox(new QSpinBox()),
        offsetZbox(new QSpinBox()),
        selete_button1(new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "")),
		//selete_button2(new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "")),
        loadmark1(new QPushButton("load mark1")),
		//loadSwc2(new QPushButton("load swc2")),
        //extractMarker(new QPushButton("extract marker")),
        hbox1(new QHBoxLayout()),
        hbox2(new QHBoxLayout()),
        hbox3(new QHBoxLayout()),
		//hbox4(new QHBoxLayout()),
		//hbox5(new QHBoxLayout()),
        layout(new QGridLayout()) 
        {
    this->setParent(parent);
    offsetXbox->setValue(0);
    offsetXbox->setRange(0, INT_MAX);
    offsetYbox->setValue(0);
    offsetYbox->setRange(0, INT_MAX);
    offsetZbox->setValue(0);
    offsetZbox->setRange(0, INT_MAX);

    hbox1->addWidget(input_mark_path_label1);
    hbox1->addWidget(input_mark_path_text1);
    hbox1->addWidget(selete_button1);

    hbox2->addWidget(offsetX);
    hbox2->addWidget(offsetXbox);
    hbox2->addWidget(offsetY);
    hbox2->addWidget(offsetYbox);
    hbox2->addWidget(offsetZ);
    hbox2->addWidget(offsetZbox);

	//hbox3->addWidget(extractMarker);
	hbox3->addWidget(loadmark1);

	//hbox4->addWidget(input_swc_path_label2);
	//hbox4->addWidget(input_swc_path_text2);
	//hbox4->addWidget(selete_button2);

	//hbox5->addWidget(loadSwc2);

    layout->addLayout(hbox1, 0, 0);
    layout->addLayout(hbox2, 1, 0);
	layout->addLayout(hbox3, 2, 0);
    //layout->addLayout(hbox4, 3, 0);
	//layout->addLayout(hbox5, 4, 0);
    setLayout(layout);
    connect(selete_button1, SIGNAL(clicked()), this, SLOT(select_Mark1()));
	//connect(selete_button2, SIGNAL(clicked()), this, SLOT(select_swc2()));

    connect(input_mark_path_text1, SIGNAL(editingFinished()), this, SLOT(update()));
	//connect(input_swc_path_text2, SIGNAL(editingFinished()), this, SLOT(update()));

    connect(offsetXbox, SIGNAL(valueChanged(int)), this, SLOT(update()));
    connect(offsetYbox, SIGNAL(valueChanged(int)), this, SLOT(update()));
    connect(offsetZbox, SIGNAL(valueChanged(int)), this, SLOT(update()));

    connect(loadmark1, SIGNAL(clicked()), this, SLOT(load1()));
	//connect(loadSwc2, SIGNAL(clicked()), this, SLOT(load2()));

    //connect(extractMarker, SIGNAL(clicked()), this, SLOT(extract()));
}


LoadMarkTeraFlyDialog::~LoadMarkTeraFlyDialog() {
    delete layout;
}

void LoadMarkTeraFlyDialog::update() {
    input_mark_path1 = input_mark_path_text1->text();
	//input_swc_path2 = input_swc_path_text2->text();
}

void LoadMarkTeraFlyDialog::select_Mark1() {
    QString select_mark_file = QFileDialog::getOpenFileName(this, "select the .marker file.", "", "*.marker *.apo");
    if (select_mark_file.isEmpty()) return;
	input_mark_path1 = select_mark_file;
	input_mark_path_text1->setText(select_mark_file);
	
}
/*
void LoadSwcTeraFlyDialog::select_swc2() {
	QString select_swc_file = QFileDialog::getOpenFileName(this, "select the .swc file.", "", "*.swc *.eswc");
	if (select_swc_file.isEmpty()) return;
	input_swc_path2 = select_swc_file;
	input_swc_path_text2->setText(select_swc_file);
}
*/
void LoadMarkTeraFlyDialog::load1(){
    QString TeraFly_name = _callback.getPathTeraFly();
    if (TeraFly_name.isEmpty()) return;
    cout<<"readMarker_file"<<endl;
    LandmarkList mark1 = readMarker_file1(input_mark_path1);
	if (offsetXbox->value() || offsetYbox->value() || offsetZbox->value()) {
		V3DLONG OriginX, OriginY, OriginZ;
		OriginX = offsetXbox->value();
		OriginY = offsetYbox->value();
		OriginZ = offsetZbox->value();
		for (LocationSimple &n : mark1) {
			n.x += OriginX;
			n.y += OriginY;
			n.z += OriginZ;
            cout<<n.x<<" "<<n.y<<" "<<n.z<<" "<<n.radius<<endl;
		}
	}
    
	_callback.setLandmarkTeraFly(mark1);
    
}

LandmarkList readMarker_file1(const QString & filename)
{
    LandmarkList tmp_list;
	QFile qf(filename);
	if (! qf.open(QIODevice::ReadOnly | QIODevice::Text))
	{
#ifndef DISABLE_V3D_MSG
		v3d_msg(QString("open file [%1] failed!").arg(filename));
#endif
		return tmp_list;
	}

	V3DLONG k=0;
    while (! qf.atEnd())
    {
		char curline[2000];
        qf.readLine(curline, sizeof(curline));
		k++;
		{
			if (curline[0]=='#' || curline[0]=='x' || curline[0]=='X' || curline[0]=='\0') continue;

			QStringList qsl = QString(curline).trimmed().split(",");
			int qsl_count=qsl.size();
			if (qsl_count<3)   continue;

			LocationSimple S;

			S.x = qsl[0].toFloat();
			S.y = qsl[1].toFloat();
			S.z = qsl[2].toFloat();
			S.radius = (qsl_count>=4) ? qsl[3].toInt() : 0;
            /*
			//S.shape = (qsl_count>=5) ? qsl[4].toInt() : 1;
			//S.name = (qsl_count>=6) ? qPrintable(qsl[5].trimmed()) : "";
			//S.comment = (qsl_count>=7) ? qPrintable(qsl[6].trimmed()) : "";

			S.color = random_rgba8(255);
			if (qsl_count>=8) S.color.r = qsl[7].toUInt();
			if (qsl_count>=9) S.color.g = qsl[8].toUInt();
			if (qsl_count>=10) S.color.b = qsl[9].toUInt();

			S.type = (S.x==-1 || S.y==-1 || S.z==-1) ? 0 : 2;

			S.on = true; //listLoc[i].on;        //true;
			S.selected = false;
            */
			tmp_list.append(S);
		}
	}

	return tmp_list;
}
/*
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
*/