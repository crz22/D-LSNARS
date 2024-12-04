#ifndef __LOAD_SWC_INTO_TERAFLY_H__
#define __LOAD_SWC_INTO_TERAFLY_H__

#include"dialog.h"

class LoadSwcTeraFlyDialog: public QDialog
{
    Q_OBJECT

public:
	QLabel *input_swc_path_label1;
    QLineEdit *input_swc_path_text1;
	QLabel *input_swc_path_label2;
	QLineEdit *input_swc_path_text2;
    QLabel *offsetX;
    QLabel *offsetY;
    QLabel *offsetZ;
    QSpinBox* offsetXbox;
    QSpinBox* offsetYbox;
    QSpinBox* offsetZbox;
    QPushButton *selete_button1;
	QPushButton *selete_button2;
    QPushButton *loadSwc1;
	QPushButton *loadSwc2;
    QPushButton *extractMarker;
    QGridLayout *layout;
    QHBoxLayout *hbox1;
    QHBoxLayout *hbox2;
    QHBoxLayout *hbox3;
	QHBoxLayout *hbox4;
	QHBoxLayout *hbox5;
    QString input_swc_path1;
	QString input_swc_path2;
private:
    V3DPluginCallback2& _callback;
public:
    explicit LoadSwcTeraFlyDialog(V3DPluginCallback2& callback, QWidget* parent = nullptr);
    ~LoadSwcTeraFlyDialog();

public Q_SLOTS:
    void update();
    void select_swc1();
	void select_swc2();
    void load1();
	void load2();
    //void extract();
};

#endif//__LOAD_SWC_INTO_TERAFLY_H__