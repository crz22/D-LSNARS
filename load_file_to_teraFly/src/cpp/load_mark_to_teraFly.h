#ifndef __LOAD_Mark_INTO_TERAFLY_H__
#define __LOAD_Mark_INTO_TERAFLY_H__

#include "dialog.h"

class LoadMarkTeraFlyDialog: public QDialog
{
    Q_OBJECT

public:
	QLabel *input_mark_path_label1;
    QLineEdit *input_mark_path_text1;
    QLabel *offsetX;
    QLabel *offsetY;
    QLabel *offsetZ;
    QSpinBox* offsetXbox;
    QSpinBox* offsetYbox;
    QSpinBox* offsetZbox;
    QPushButton *selete_button1;
    QPushButton *loadmark1;
    QGridLayout *layout;
    QHBoxLayout *hbox1;
    QHBoxLayout *hbox2;
    QHBoxLayout *hbox3;
    QString input_mark_path1;

private:
    V3DPluginCallback2& _callback;
public:
    explicit LoadMarkTeraFlyDialog(V3DPluginCallback2& callback, QWidget* parent = nullptr);
    ~LoadMarkTeraFlyDialog();

public Q_SLOTS:
    void update();
    void select_Mark1();
    void load1();
};

LandmarkList readMarker_file1(const QString & filename);

#endif//__LOAD_Mark_INTO_TERAFLY_H__