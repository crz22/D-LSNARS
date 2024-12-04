#ifndef __DIALOG_H__
#define __DIALOG_H__

#include <QtGui>
#include "v3d_interface.h"
#include <QWidget>
#include <iostream>
#include <QApplication>
#include <QDateTime>
//#include "surf_objs.h"
#include "basic_surf_objs.h"
#include <qtpropertybrowser.h>
#include <qtpropertymanager.h>
#include <qtvariantproperty.h>
#include <qttreepropertybrowser.h>
#include "soma_detection.h"
//#include "neuron_reconstruction.h"
#define PI 3.1415926

using namespace std;

//Dialog functions
class Detection_Dialog : public QDialog
{
	Q_OBJECT
public:
    QGridLayout *OverallLayout;
    
    //file input and output
    QGroupBox *groupBox_filepath;
    QGridLayout *Layout_filepath;
    
    QHBoxLayout *HBox_input_path;
    QLabel *input_path_label;
    QLineEdit *input_path_text;
    QPushButton *input_path_button;
    
    QHBoxLayout *HBox_save_path;
    QLabel *save_path_label;
    QLineEdit *save_path_text;
    QPushButton *save_path_button;

    QHBoxLayout *HBox_pytorch_path;
    QLabel *pytorch_path_label;
    QLineEdit *pytorch_path_text;
    QPushButton *pytorch_path_button;

    QString save_path_value;
    QString input_path_value;
    QString pytorch_path_value;
   
    //soma_detection parameter
    QGroupBox *groupBox_detection_parameters;
    QGridLayout *Layout_detection_parameters;

    QDoubleSpinBox *CE; //contrast enhancement
    QSpinBox *SMA; //SOMA MIN AERA
    QSpinBox *DB_eps; //DBSCAN eps
    QSpinBox *DB_MP;  //DBSCAN Minpts

    QLabel *CE_label; //contrast enhancement
    QLabel *SMA_label; //SOMA MIN AERA
    QLabel *DB_eps_label; //DBSCAN eps
    QLabel *DB_MP_label;  //DBSCAN Minpts

    QHBoxLayout *HBox_CE;
    QHBoxLayout *HBox_SMA;
    QHBoxLayout *HBox_DB_eps;
    QHBoxLayout *HBox_DB_MP;

    float CE_value, SMA_value, DB_eps_value, DB_MP_value;

    //start button
    QPushButton *Start_button;
    Soma_Detection *soma_detection;

    
private:
    V3DPluginCallback2& _callback;
    //SomaDetection* function1;

public:
    explicit Detection_Dialog(V3DPluginCallback2& callback, QWidget* parent = nullptr);
    ~Detection_Dialog();   

private slots:
    //private slots
    void Select_input_path();
    void Select_save_path();
    void Select_pytorch_path();
    void update_input_path();
    void update_save_path();
    void update_pytorch_path();
    void update_CE();
    void update_SMA();
    void update_DB_eps();
    void update_DB_MP();
    void Start_soma_detection();
    //void Neuron_reconstruct();
};

#endif //__DIALOG_H__
