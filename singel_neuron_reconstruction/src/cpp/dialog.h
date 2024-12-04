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
#include "neuron_reconstruction.h"

#define PI 3.1415926
using namespace std;

enum {SPEDNR=0, APP2};
enum {DTANET=0, UNET3D};

//Dialog functions
class Reconstruction_Dialog : public QDialog
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

    QHBoxLayout *HBox_marker_path;
    QLabel *marker_path_label;
    QLineEdit *marker_path_text;
    QPushButton *marker_path_button;
    
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
    QString marker_path_value;
    QString pytorch_path_value;
    
    //soma_detection parameter
    QGroupBox *groupBox_reconstruct_parameters;
    QGridLayout *Layout_reconstruct_parameters;

    QHBoxLayout *HBox_Method;
    QComboBox *traceMethod;
    QLabel *traceMethod_label;
    QString traceMethod_value;

    QComboBox *segmentMethod;
    QLabel *segmentMethod_label;
    QString segmentMethod_value;
    
    QHBoxLayout *HBox_blockSize;
    QLabel *blockSize_label; 
    QSpinBox *blockSize_input;
    int blockSize_value;

    QHBoxLayout *HBox_margin_lamd; //Overlap ratio of adjacent blocks
    QLabel *margin_lamd_label; 
    QDoubleSpinBox *margin_lamd_input;
    float margin_lamd_value;

    QHBoxLayout *HBox_MaxpointNUM;
    QLabel *MaxpointNUM_label; 
    QSpinBox *MaxpointNUM_input;
    long long MaxpointNUM_value;

    QHBoxLayout *HBox_min_branch_length;
    QLabel *min_branch_length_label; 
    QSpinBox *min_branch_length_input;
    int min_branch_length_value;

    /**/
    //start button
    QPushButton *Start_button;
    Neuron_Reconstruction *neuron_reconstruction;


private:
    V3DPluginCallback2& _callback;
    void SETfilepath(V3DPluginCallback2& callback);
    void SETparameter();

public:
    explicit Reconstruction_Dialog(V3DPluginCallback2& callback, QWidget* parent = nullptr);
    ~Reconstruction_Dialog();   

private slots:
    //private slots
    
    void Select_input_path();
    void Select_marker_path();
    void Select_save_path();
    void Select_pytorch_path();
    void update_input_path();
    void update_marker_path();
    void update_save_path();
    void update_pytorch_path();
    void SelecttraceMethod();
    void SelectsegmentMethod();
    
    void update_blockSize();
    void update_margin_lamd();
    void update_MaxpointNUM();
    void update_min_branch_length();
    /**/ 
    void Start_neuron_reconstruction();   
};

#endif //__DIALOG_H__


